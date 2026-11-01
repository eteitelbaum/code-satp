"""How big SHRUG's villages are, and how often OpenStreetMap's village points fall in the wrong one (GitHub #26).

In the sample's districts:
  1. Village size: area of SHRUG's village polygons, and the width of a circle of the same area;
     by state (2011 states: Andhra Pradesh includes Telangana). Subdistrict size, and villages and
     towns per subdistrict.
  2. OpenStreetMap vs SHRUG: for each named OpenStreetMap village or hamlet point, does it fall in
     the SHRUG village of the same name (spelling similarity >= 85), in a different village with a
     same-name village within 5 km (the wrong village), or neither? For the wrong ones, how far the
     point is from the same-name village's boundary. By state, and the distance from the point to
     the SHRUG village's centroid.
  3. With --provenance: the same split by OpenStreetMap's source tag, edit count, and last edit
     year, read from the Geofabrik file (slow; see build_osm.py).

Distances are approximate (degrees times 111 km). Per-point results go to data/osm_vs_shrug.csv.

Run from the repository root:  python location-labeling/boundary_checks.py [--provenance]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from rapidfuzz import fuzz
from scipy.spatial import cKDTree

DATA = Path(__file__).parent / "data"
KM = 111.32
SAME_NAME = 85
NEARBY_KM = 5.0


def name_key(x):
    return re.sub(r"[^a-z]", "", re.sub(r"\(.*?\)", "", str(x).lower()))


def pct(s):
    return f"{s.mean():.0%} of {len(s)}"


def in_sample(df, dists, district="district_id"):
    return df[[(s, d) in dists for s, d in zip(df.state_id, df[district])]]


def villages(dists):
    """SHRUG villages in the sample's districts, with polygons and areas in km²."""
    pl = in_sample(pd.read_parquet(DATA / "places.parquet"), dists)
    pl = pl[pl.place_type == "village"].reset_index(drop=True)
    wkb = pd.read_parquet(DATA / "place_shapes.parquet").set_index("shrid2").wkb
    pl["geom"] = shapely.from_wkb(wkb.loc[pl.shrid2].values)
    pl["km2"] = shapely.area(pl.geom) * KM * KM * np.cos(np.radians(pl.lat))
    pl["key"] = pl.name.map(name_key)
    return pl


def sizes(pl, dists):
    q = [.1, .25, .5, .75, .9]
    a = pl.km2.quantile(q)
    print(f"Villages in the sample's {len(dists)} districts: {len(pl)}")
    print("  area, km² (10/25/50/75/90%):          ", a.round(2).tolist())
    print("  width of a circle that size, km:      ", (2 * np.sqrt(a / np.pi)).round(2).tolist())
    for s, g in pl.groupby("state_name"):
        print(f"  {s:<15} {len(g):>6} villages, median {g.km2.median():.2f} km²")
    sd = in_sample(pd.read_parquet(DATA / "subdistrict_shapes.parquet"), dists)
    g = shapely.from_wkb(sd.wkb.values)
    area = shapely.area(g) * KM * KM * np.cos(np.radians(shapely.get_y(shapely.centroid(g))))
    print(f"Subdistricts: {len(sd)}, area km² (25/50/75%):", np.percentile(area, [25, 50, 75]).round(0).tolist())
    n = in_sample(pd.read_parquet(DATA / "places.parquet"), dists).groupby(
        ["state_id", "district_id", "subdistrict_id"]).size()
    print("  villages and towns per subdistrict (25/50/75%):", n.quantile([.25, .5, .75]).tolist())


def osm_vs_shrug(pl):
    """Each named OpenStreetMap village or hamlet point: same-name village, wrong village, or neither."""
    by_id = pl.set_index("shrid2")
    tree = cKDTree(np.c_[pl.lon * np.cos(np.radians(pl.lat)), pl.lat] * KM)
    centroids = shapely.centroid(pl.geom.values)
    cen = dict(zip(pl.shrid2, zip(shapely.get_x(centroids), shapely.get_y(centroids))))
    osm = pd.read_parquet(DATA / "osm_places.parquet")
    o = osm[osm.kind.isin(["village", "hamlet"]) & (osm.name == osm.primary) & osm.shrid2.isin(by_id.index)]
    o = o.drop_duplicates("osm_id").copy()
    o["key"] = o.name.map(name_key)
    o = o[o.key.str.len() >= 4]

    rows = []
    for r in o.itertuples():
        kx = KM * np.cos(np.radians(r.lat))
        if fuzz.ratio(r.key, by_id.at[r.shrid2, "key"]) >= SAME_NAME:
            match, past = r.shrid2, 0.0
            outcome = "same-name village"
        else:
            near = [pl.iloc[i] for i in tree.query_ball_point([r.lon * kx, r.lat * KM], NEARBY_KM)]
            near = [c for c in near if c.shrid2 != r.shrid2 and fuzz.ratio(r.key, c.key) >= SAME_NAME]
            if not near:
                rows.append((r.osm_id, r.kind, by_id.at[r.shrid2, "state_name"], "no same-name village nearby", "", np.nan, np.nan))
                continue
            p = shapely.Point(r.lon, r.lat)
            c = min(near, key=lambda c: c.geom.distance(p))
            match, past = c.shrid2, c.geom.distance(p) * KM
            outcome = "wrong village"
        cx, cy = cen[match]
        rows.append((r.osm_id, r.kind, by_id.at[r.shrid2, "state_name"], outcome, match, past, np.hypot((r.lon - cx) * kx, (r.lat - cy) * KM)))
    return pd.DataFrame(rows, columns=["osm_id", "kind", "state", "outcome", "same_name_shrid2",
                                       "km_past_boundary", "km_to_centroid"])


def report(res):
    print()
    print("OpenStreetMap points vs SHRUG villages:")
    print(pd.crosstab(res.outcome, res.kind, margins=True))
    m = res[res.outcome != "no same-name village nearby"]
    wrong = m.outcome == "wrong village"
    print("\nShare in the wrong village, of points with a same-name village nearby, by kind:")
    print(wrong.groupby(m.kind).apply(pct).to_string())
    print("\nVillage points in the wrong village, by state:")
    v = m[m.kind == "village"]
    print((v.outcome == "wrong village").groupby(v.state).apply(pct).to_string())
    for k, g in m[wrong].groupby("kind"):
        print(f"\n{k}: km past the same-name village's boundary (25/50/75/90%):",
              g.km_past_boundary.quantile([.25, .5, .75, .9]).round(2).tolist(),
              f" within 0.5 km: {(g.km_past_boundary <= 0.5).mean():.0%}")
    for k, g in m[m.kind == "village"].groupby("outcome"):
        print(f"village, {k}: km to the SHRUG centroid (25/50/75/90%):",
              g.km_to_centroid.quantile([.25, .5, .75, .9]).round(2).tolist())


def provenance(res):
    """Source tag, edit count, and last edit year of the village nodes, from the Geofabrik file."""
    import osmium

    v = res[(res.kind == "village") & (res.outcome != "no same-name village nearby") & res.osm_id.str.startswith("node/")]
    want = {int(x.split("/")[1]) for x in v.osm_id}
    meta = {}
    nodes = osmium.FileProcessor(str(DATA / "osm" / "india-latest.osm.pbf"), osmium.osm.NODE)
    for n in nodes.with_filter(osmium.filter.KeyFilter("place")):
        if n.id in want:
            meta[f"node/{n.id}"] = dict(version=n.version, year=n.timestamp.year, source=n.tags.get("source", ""))
    v = v.join(pd.DataFrame.from_dict(meta, orient="index"), on="osm_id")
    src = v.source.fillna("").str.lower()
    v["src"] = src.str.extract(r"(pmgsy|bharat|census|gns|bing|survey|esri|maxar|local|wikipedia|lgd)",
                               expand=False).fillna(pd.Series(np.where(src == "", "(none)", "other"), index=v.index))
    wrong = v.outcome == "wrong village"
    print(f"\nProvenance of {len(v)} village nodes:")
    print("  wrong village, by source tag:")
    print(wrong.groupby(v.src).apply(pct).to_string())
    print("  by number of versions:")
    vgrp = pd.cut(v.version, [0, 1, 2, 5, 10**6], labels=["1 (never edited)", "2", "3-5", "6+"])
    print(wrong.groupby(vgrp, observed=True).apply(pct).to_string())
    print("  last edited, year (25/50/75%):", v.year.quantile([.25, .5, .75]).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provenance", action="store_true", help="also read the Geofabrik file for edit history (slow)")
    args = ap.parse_args()

    sample = pd.read_csv(DATA / "sample.csv", dtype=str)
    dists = set(zip(sample.state_id, sample.district_id_2011))
    pl = villages(dists)
    sizes(pl, dists)
    res = osm_vs_shrug(pl)
    res.to_csv(DATA / "osm_vs_shrug.csv", index=False)
    report(res)
    if args.provenance:
        provenance(res)


if __name__ == "__main__":
    main()
