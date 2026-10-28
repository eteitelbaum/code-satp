"""Named places from OpenStreetMap for the study states (GitHub #16).

suggest.py matches place names against this table with the same spelling measure it uses
for SHRUG, so "Nishanguda" can find "Nisanguda" (OpenStreetMap's own search matches whole
words only). Keeps settlements (city, town, village, hamlet, locality, ...), forests,
protected areas (sanctuaries, national parks, reserves), hills, peaks, ridges, and valleys,
under every Latin-script name they carry (name, name:en, alt_name, old_name, ...), plus a
short form of forest and sanctuary names ("Karlapat WLS" -> "Karlapat"). Each is tied to
the SHRUG village and subdistrict it falls in. Outlines of forests and protected areas are
kept too, for protocol rule 4 (a point in the part of a forest within a jurisdiction).

Download India from Geofabrik first (about 1.7 GB):
    curl -L -o location-labeling/data/osm/india-latest.osm.pbf https://download.geofabrik.de/asia/india-latest.osm.pbf

Run from the repository root:  python location-labeling/build_osm.py

Output (location-labeling/data/):
    osm_places.parquet  one row per place and name, with a point inside it
    osm_areas.parquet   outlines and sizes of forests and protected areas
Data © OpenStreetMap contributors, available under the Open Database License (ODbL).
"""

import re
from pathlib import Path

import numpy as np
import osmium
import pandas as pd
import shapely

DATA = Path(__file__).parent / "data"
PBF = DATA / "osm" / "india-latest.osm.pbf"
OUT = DATA / "osm_places.parquet"
AREAS = DATA / "osm_areas.parquet"

PLACES = {"city", "town", "village", "hamlet", "isolated_dwelling", "locality", "neighbourhood", "suburb", "quarter", "farm"}
NATURAL = {"wood": "forest", "peak": "peak", "hill": "hill", "ridge": "ridge", "valley": "valley"}
AREA_KINDS = {"forest", "protected area"}  # kinds whose outlines are kept
# words that describe a forest or hill rather than name it
GENERIC = re.compile(r"\b(wildlife sanctuary|sanctuary|wls|reserved? forests?|protected forests?|rf|pf|forests?|"
                     r"national park|tiger reserve|elephant reserve|reserve|hill range|range|hills?|jungle|woods?)\b",
                     re.IGNORECASE)
NAME_KEYS = ["name", "name:en", "alt_name", "alt_name:en", "old_name", "official_name", "loc_name", "short_name"]


def kind(tags):
    if tags.get("place") in PLACES:
        return tags["place"]
    if tags.get("natural") in NATURAL:
        return NATURAL[tags["natural"]]
    if tags.get("landuse") == "forest" or tags.get("boundary") == "forest":
        return "forest"
    if tags.get("leisure") == "nature_reserve" or tags.get("boundary") in ("protected_area", "national_park"):
        return "protected area"
    return None


def names(tags, k):
    out = []
    for key in NAME_KEYS:
        for n in (tags.get(key) or "").split(";"):
            n = n.strip()
            if len(re.sub(r"[^a-z]", "", n.lower())) >= 3 and n not in out:  # Latin script only
                out.append(n)
    if k in AREA_KINDS | {"hill", "peak", "ridge", "valley"}:  # "Karlapat WLS" is also found as "Karlapat"
        for n in list(out):
            short = re.sub(r"\s+", " ", GENERIC.sub(" ", n)).strip(" .,-")
            if len(re.sub(r"[^a-z]", "", short.lower())) >= 3 and short not in out:
                out.append(short)
    return out


def main():
    wkb = osmium.geom.WKBFactory()
    rows, outlines = [], []
    fp = (osmium.FileProcessor(str(PBF)).with_locations().with_areas()
          .with_filter(osmium.filter.KeyFilter("place", "natural", "landuse", "boundary", "leisure")))
    for obj in fp:
        k = kind(obj.tags)
        if not k:
            continue
        ns = names(obj.tags, k)
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
                if k in AREA_KINDS:
                    outlines.append({"osm_id": oid, "geom": geom})
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

    # outlines of the forests and protected areas kept above, simplified to about 50 m
    areas = pd.DataFrame(outlines).drop_duplicates("osm_id")
    areas = areas[areas.osm_id.isin(osm.osm_id)].reset_index(drop=True)
    geoms = shapely.simplify(areas["geom"].values, 0.0005, preserve_topology=True)
    lat = shapely.get_y(shapely.centroid(geoms))
    areas["area_km2"] = shapely.area(geoms) * 111.32 ** 2 * np.cos(np.radians(lat))
    areas["wkb"] = shapely.to_wkb(geoms)
    areas[["osm_id", "area_km2", "wkb"]].to_parquet(AREAS, index=False)
    print(f"{len(areas):,} forest and protected-area outlines")
    print(f"{len(osm):,} in the study states ({osm.shrid2.ne('').mean():.0%} inside a SHRUG village), "
          f"{len(out):,} names; by kind:")
    print(osm["kind"].value_counts().to_string())


if __name__ == "__main__":
    main()
