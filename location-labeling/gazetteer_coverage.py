"""How often each gazetteer lists the coders' village (GitHub #22).

For each sample incident with a coded village, is there a place with that name (spelling
similarity >= 0.85, after dropping words such as "village") inside the coded 2011 district,
in SHRUG, OpenStreetMap, and GeoNames? The coders' fields are a rough reference, not gold
labels; the comparison across gazetteers is the point.

Download GeoNames' India file first (CC BY 4.0):
    curl -L -o location-labeling/data/geonames/IN.zip https://download.geonames.org/export/dump/IN.zip
    unzip location-labeling/data/geonames/IN.zip -d location-labeling/data/geonames

Run from the repository root:  python location-labeling/gazetteer_coverage.py validation
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import shapely
from rapidfuzz import process

from suggest import GENERIC, norm, spelling

DATA = Path(__file__).parent / "data"
SAME_NAME = 0.85
GEONAMES_COLUMNS = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "feature_class", "feature_code",
                    "country", "cc2", "admin1", "admin2", "admin3", "admin4", "population", "elevation", "dem",
                    "timezone", "modified"]


def geonames(subs):
    """GeoNames places in India, one row per name, tagged with the SHRUG district they fall in."""
    g = pd.read_csv(DATA / "geonames" / "IN.txt", sep="\t", names=GEONAMES_COLUMNS, dtype=str, quoting=3,
                    keep_default_na=False)
    tree = shapely.STRtree(shapely.from_wkb(subs["wkb"].values))
    pi, si = tree.query(shapely.points(g["lon"].astype(float), g["lat"].astype(float)), predicate="intersects")
    district = pd.Series(subs["district_id"].astype(str).values[si], index=pi).groupby(level=0).first()
    g = g.iloc[district.index].assign(district_id=district.values)
    g["names"] = [[n, a] + [x for x in alt.split(",") if x] for n, a, alt in zip(g["name"], g["asciiname"], g["alternatenames"])]
    return g.explode("names").rename(columns={"names": "place"})[["place", "district_id", "feature_class"]]


def by_district(df, col):
    df = df.assign(key=[norm(x) for x in df[col]])
    df = df[df.key.str.len() >= 3]
    return df.groupby(df["district_id"].astype(str))["key"].apply(lambda s: sorted(set(s))).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["validation", "test"])
    args = ap.parse_args()

    sample = pd.read_csv(DATA / "sample.csv", dtype=str).fillna("")
    sample = sample[(sample.split == args.split) & (sample.village_name.str.strip() != "")]
    subs = pd.read_parquet(DATA / "subdistrict_shapes.parquet")
    gn = geonames(subs)
    lists = {
        "SHRUG": by_district(pd.read_parquet(DATA / "places.parquet"), "name"),
        "OpenStreetMap": by_district(pd.read_parquet(DATA / "osm_places.parquet"), "name"),
        "GeoNames": by_district(gn, "place"),
        "GeoNames, populated places": by_district(gn[gn.feature_class == "P"], "place"),
    }

    found = {k: set() for k in lists}
    for r in sample.itertuples():
        for name in r.village_name.split(","):  # a few incidents code two villages
            query = norm(re.sub(GENERIC, " ", name, flags=re.I))
            if len(query) < 3:
                continue
            for k, names in lists.items():
                if process.extractOne(query, names.get(str(r.district_id_2011), []), scorer=spelling, score_cutoff=SAME_NAME):
                    found[k].add(r.incident_number)

    n = len(sample)
    print(f"{args.split}: {n} incidents with a coded village; a place with that name inside the coded district in")
    for k, s in found.items():
        print(f"  {k:28} {len(s):4} ({len(s) / n:.0%})")
    shrug_osm = found["SHRUG"] | found["OpenStreetMap"]
    print(f"  {'SHRUG or OpenStreetMap':28} {len(shrug_osm):4} ({len(shrug_osm) / n:.0%})")
    print(f"  GeoNames only (in neither SHRUG nor OpenStreetMap): {len(found['GeoNames'] - shrug_osm)}")
    print(f"  SHRUG or OpenStreetMap only (not in GeoNames):     {len(shrug_osm - found['GeoNames'])}")
    print(f"  in none of the three: {n - len(shrug_osm | found['GeoNames'])}")


if __name__ == "__main__":
    main()
