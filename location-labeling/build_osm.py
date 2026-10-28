"""Named places from OpenStreetMap for the study states (GitHub #16).

suggest.py matches place names against this table with the same spelling measure it uses
for SHRUG, so "Nishanguda" can find "Nisanguda" (OpenStreetMap's own search matches whole
words only). Keeps settlements (city, town, village, hamlet, locality, ...), forests,
hills, peaks, ridges, and valleys, under every Latin-script name they carry (name, name:en,
alt_name, old_name, ...). Each is tied to the SHRUG village and subdistrict it falls in.

Download India from Geofabrik first (about 1.7 GB):
    curl -L -o location-labeling/data/osm/india-latest.osm.pbf https://download.geofabrik.de/asia/india-latest.osm.pbf

Run from the repository root:  python location-labeling/build_osm.py

Output: location-labeling/data/osm_places.parquet, one row per place and name.
Data © OpenStreetMap contributors, available under the Open Database License (ODbL).
"""

import re
from pathlib import Path

import osmium
import pandas as pd
import shapely

DATA = Path(__file__).parent / "data"
PBF = DATA / "osm" / "india-latest.osm.pbf"
OUT = DATA / "osm_places.parquet"

PLACES = {"city", "town", "village", "hamlet", "isolated_dwelling", "locality", "neighbourhood", "suburb", "quarter", "farm"}
NATURAL = {"wood": "forest", "peak": "peak", "hill": "hill", "ridge": "ridge", "valley": "valley"}
NAME_KEYS = ["name", "name:en", "alt_name", "alt_name:en", "old_name", "official_name", "loc_name", "short_name"]


def kind(tags):
    if tags.get("place") in PLACES:
        return tags["place"]
    if tags.get("natural") in NATURAL:
        return NATURAL[tags["natural"]]
    if tags.get("landuse") == "forest" or tags.get("boundary") == "forest":
        return "forest"
    return None


def names(tags):
    out = []
    for k in NAME_KEYS:
        for n in (tags.get(k) or "").split(";"):
            n = n.strip()
            if len(re.sub(r"[^a-z]", "", n.lower())) >= 3 and n not in out:  # Latin script only
                out.append(n)
    return out


def main():
    wkb = osmium.geom.WKBFactory()
    rows = []
    fp = (osmium.FileProcessor(str(PBF)).with_locations().with_areas()
          .with_filter(osmium.filter.KeyFilter("place", "natural", "landuse", "boundary")))
    for obj in fp:
        k = kind(obj.tags)
        if not k:
            continue
        ns = names(obj.tags)
        if not ns:
            continue
        try:
            if obj.is_node():
                point = (obj.location.lon, obj.location.lat)
                oid = f"node/{obj.id}"
            elif obj.is_area():
                geom = shapely.from_wkb(wkb.create_multipolygon(obj))
                p = shapely.point_on_surface(geom)
                point = (p.x, p.y)
                oid = f"{'way' if obj.from_way() else 'relation'}/{obj.orig_id()}"
            elif obj.is_way() and not obj.is_closed():  # ridges and valleys are drawn as lines
                coords = [(n.lon, n.lat) for n in obj.nodes]
                point = coords[len(coords) // 2]
                oid = f"way/{obj.id}"
            else:
                continue  # closed ways come through again as areas
        except (osmium.InvalidLocationError, RuntimeError, shapely.errors.GEOSException):
            continue
        rows.append({"osm_id": oid, "kind": k, "primary": ns[0], "names": ns, "lon": point[0], "lat": point[1]})
    osm = pd.DataFrame(rows).drop_duplicates("osm_id")
    print(f"{len(osm):,} named places in India")

    # keep the study states: tie each place to the subdistrict, and the SHRUG village, it falls in
    subs = pd.read_parquet(DATA / "subdistrict_shapes.parquet")
    pts = shapely.points(osm["lon"].values, osm["lat"].values)
    tree = shapely.STRtree(shapely.from_wkb(subs["wkb"].values))
    pi, si = tree.query(pts, predicate="intersects")
    first = pd.Series(si, index=pi).groupby(level=0).first()
    osm = osm.iloc[first.index].reset_index(drop=True)
    osm[["state_id", "district_id", "subdistrict_id"]] = subs.iloc[first.values][["state_id", "district_id", "subdistrict_id"]].values
    villages = pd.read_parquet(DATA / "place_shapes.parquet")
    vtree = shapely.STRtree(shapely.from_wkb(villages["wkb"].values))
    pi, vi = vtree.query(shapely.points(osm["lon"].values, osm["lat"].values), predicate="intersects")
    vfirst = pd.Series(vi, index=pi).groupby(level=0).first()
    osm["shrid2"] = ""
    osm.loc[vfirst.index, "shrid2"] = villages.iloc[vfirst.values]["shrid2"].values

    out = osm.explode("names").rename(columns={"names": "name"}).reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(f"{len(osm):,} in the study states ({osm.shrid2.ne('').mean():.0%} inside a SHRUG village), "
          f"{len(out):,} names; by kind:")
    print(osm["kind"].value_counts().to_string())


if __name__ == "__main__":
    main()
