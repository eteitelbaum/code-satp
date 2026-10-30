"""Mordecai 3 as is on the sample (GitHub #22, step 0).

Mordecai finds place names with spaCy, looks them up in a GeoNames index, and ranks the
candidates with a neural model, giving each place a calibrated score (or "no match").
Version 3.5 no longer picks which place is where the event happened; it returns them all.
This script runs it on the same text suggest.py reads (the coded summary, then the SATP
entry) and reports, before any gold labels, how often it places something below the
district, finds the coders' village, and lands inside the coded district.

Setup (once; Mordecai's dependencies stay out of the main environment):

    cd location-labeling/data/mordecai
    python -m venv venv && venv/bin/pip install mordecai3==3.5.0
    venv/bin/python -m spacy download en_core_web_trf
    venv/bin/mordecai3 index fetch --dir .      # prebuilt GeoNames index, 1.5 GB
    docker run -d --name mordecai-es -p 127.0.0.1:9200:9200 -e discovery.type=single-node \
        -e ES_JAVA_OPTS="-Xms2g -Xmx2g" -v $PWD/geonames_index:/usr/share/elasticsearch/data elasticsearch:7.10.1

Two steps, each in its own environment:

    # run: Mordecai's environment, with Elasticsearch running
    location-labeling/data/mordecai/venv/bin/python location-labeling/mordecai_baseline.py run validation

    # report: the main environment
    venv/bin/python location-labeling/mordecai_baseline.py report validation

Output: location-labeling/data/mordecai/<split>.jsonl, one line per incident.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = DATA / "mordecai"
TOP_K = 5  # candidates kept per place, for candidate coverage (--top-k up to 100 to see what the search retrieved)
# --strip: words dropped from the end of spaCy's place names before the GeoNames lookup, as Mordecai
# already drops a leading "the" ("Kulhi village" -> "Kulhi", "Bacheli Police Station" -> "Bacheli")
TRAILING_GENERIC = {"village", "villages", "police", "station", "ps", "outpost", "camp", "district", "block", "mandal",
                    "tehsil", "panchayat", "area", "areas", "forest", "forests", "reserve", "reserved", "rf", "jungle",
                    "jungles", "hill", "hills", "range", "valley"}
SAME_NAME = 0.85  # spelling similarity for "the coders' village"
NOT_BELOW = {"PCLI", "ADM1", "ADM2"}  # country, state, district


def output(split, strip, top_k, fuzzy):
    return OUT / (f"{split}{'_stripped' if strip else ''}{f'_k{top_k}' if top_k != TOP_K else ''}"
                  f"{f'_fuzzy{fuzzy}' if fuzzy else ''}.jsonl")


def load(split):
    sample = pd.read_csv(DATA / "sample.csv", dtype=str).fillna("")
    sample = sample[sample.split == split].assign(o=lambda d: d["order"].astype(int)).sort_values("o")
    links = pd.read_csv(DATA / "satp_links.csv", dtype=str).dropna(subset=["entry_id"]).fillna("")
    entries = pd.read_csv(DATA / "satp_entries.csv", dtype=str)
    text = dict(zip(zip(entries["month"], entries["entry_id"]), entries["text"]))
    satp = {r.incident_number: [text[(r.entry_month, r.entry_id)]] + [text[tuple(a.split())] for a in r.also.split(";") if a]
            for r in links.itertuples()}
    return sample, satp


def run(split, limit, strip, top_k, fuzzy):
    from mordecai3 import Geoparser, geoparse

    if fuzzy:  # by default the lookup is an exact-phrase search, with a fuzzy retry only when it finds nothing
        lookup = geoparse.add_es_data_batch
        geoparse.add_es_data_batch = lambda *a, **kw: lookup(*a, **{**kw, "fuzzy": fuzzy})

    if strip:
        trim = geoparse.trim_span_tokens

        def trim_generic(tokens):
            tokens = trim(tokens)
            while len(tokens) > 1 and tokens[-1].text.lower() in TRAILING_GENERIC:
                tokens = tokens[:-1]
            return tokens

        geoparse.trim_span_tokens = trim_generic
    sample, satp = load(split)
    if limit:
        sample = sample.head(limit)
    done = set()
    path = output(split, strip, top_k, fuzzy)
    if path.exists():
        done = {json.loads(line)["incident_number"] for line in path.open()}
    todo = sample[~sample.incident_number.isin(done)]
    print(f"{len(todo)} incidents to run ({len(done)} already done)")
    if todo.empty:
        return
    texts = [r.incident_summary + ("\n\n" + "\n\n".join(satp[r.incident_number]) if r.incident_number in satp else "")
             for r in todo.itertuples()]
    geo = Geoparser()
    results = geo.geoparse_batch(texts, known_country="IND", top_k=top_k, show_progress=True)
    with path.open("a") as f:
        for r, res in zip(todo.itertuples(), results):
            f.write(json.dumps({"incident_number": r.incident_number, "summary_chars": len(r.incident_summary),
                                "places": res["geolocated_ents"]}) + "\n")


def report(split, strip, top_k, fuzzy):
    import re

    import shapely

    from suggest import GENERIC, norm, spelling

    sample, _ = load(split)
    rows = [json.loads(line) for line in output(split, strip, top_k, fuzzy).open()]
    res = pd.DataFrame([{"incident_number": r["incident_number"], "in_summary": p["start_char"] < r["summary_chars"], **p}
                        for r in rows for p in r["places"]])
    res = res.merge(sample[["incident_number", "state", "district", "district_id_2011", "village_name"]], on="incident_number")
    # spaCy's spans often carry words such as village or police station ("Kulhi village"); compare names without them
    res["bare"] = [re.sub(r"\s+", " ", re.sub(GENERIC, " ", s, flags=re.I)).strip() for s in res.search_name]
    n = len(rows)

    # which SHRUG district a point falls in
    subs = pd.read_parquet(DATA / "subdistrict_shapes.parquet")
    tree = shapely.STRtree(shapely.from_wkb(subs["wkb"].values))

    def district_of(lon, lat):
        pi, si = tree.query(shapely.points(lon, lat), predicate="intersects")
        hit = pd.Series(subs["district_id"].astype(str).values[si], index=pi).groupby(level=0).first()
        return hit.reindex(range(len(lon))).fillna("").values

    placed = res[~res.no_match.astype(bool)].copy()
    placed["in_district"] = district_of(placed["lon"].astype(float).values, placed["lat"].astype(float).values) \
        == placed["district_id_2011"].astype(str).values

    # below the district: not a country, state, or district, and not named after the coded state or district
    named_area = [max(spelling(norm(p.search_name), norm(p.district)), spelling(norm(p.search_name), norm(p.state))) >= 0.9
                  for p in placed.itertuples()]
    below = placed[~placed.feature_code.isin(NOT_BELOW) & ~pd.Series(named_area, index=placed.index)]

    def share(k, of):
        return f"{k} of {of} ({k / of:.0%})" if of else "0"

    print(f"Mordecai 3.5 {'with generic words stripped' if strip else 'as is'}{f', fuzzy search {fuzzy}' if fuzzy else ''}, {split}: {n} incidents, {len(res)} place names found, {len(placed)} placed, "
          f"{int(res.no_match.astype(bool).sum())} no match")
    print(f"  incidents with a place below the district: {share(below.incident_number.nunique(), n)}"
          f" (from the coded summary alone: {share(below[below.in_summary].incident_number.nunique(), n)})")
    confident = below[below.score.astype(float) >= 0.7]
    print(f"  ... with score >= 0.7: {share(confident.incident_number.nunique(), n)}")
    print(f"  places below the district inside the coded district: {share(int(below.in_district.sum()), len(below))}")

    # the coders' village: found as a name, placed, and placed inside the coded district
    coded = sample[sample.village_name.str.strip() != ""]
    found = res[[spelling(norm(p.bare), norm(p.village_name)) >= SAME_NAME if p.village_name else False
                 for p in res.itertuples()]]
    found_placed = placed.loc[placed.index.intersection(found.index)]
    # was any of the top candidates inside the coded district? (GeoNames has a fitting place; the ranker chose another)
    cands = pd.DataFrame([{"i": i, "district_id_2011": str(p.district_id_2011), **c}
                          for i, p in found.iterrows() if isinstance(p.candidates, list) for c in p.candidates])
    cand_ok, cand_named, best_rank, filtered = set(), set(), pd.Series(dtype=int), set()
    if len(cands):
        cands["rank"] = cands.groupby("i").cumcount() + 1
        cands["in_district"] = district_of(cands["lon"].astype(float).values, cands["lat"].astype(float).values) \
            == cands["district_id_2011"].values
        cand_ok = set(found.loc[cands[cands.in_district].i.unique(), "incident_number"])
        # the coded village itself: an in-district candidate with the coded name (what a district filter would keep)
        cands["named"] = [spelling(norm(c), norm(found.at[i, "village_name"])) >= SAME_NAME for i, c in zip(cands.i, cands.name)]
        hit = cands[cands.in_district & cands.named]
        cand_named = set(found.loc[hit.i.unique(), "incident_number"])
        best_rank = hit.assign(inc=found.loc[hit.i, "incident_number"].values).groupby("inc")["rank"].min()
        # a district filter on Mordecai: each name's best-scoring candidate inside the coded district
        top_in = cands[cands.in_district].sort_values("score", ascending=False).drop_duplicates("i")
        filtered = set(found.loc[top_in[top_in.named].i, "incident_number"])
    print(f"  incidents with a coded village: {len(coded)}")
    print(f"    village name found by spaCy (words such as 'village' set aside): {share(found.incident_number.nunique(), len(coded))}")
    print(f"    ... a top-{top_k} GeoNames candidate inside the coded district: {share(len(cand_ok), len(coded))}")
    print(f"    ... a top-{top_k} candidate inside the coded district with the coded name: {share(len(cand_named), len(coded))}")
    if len(best_rank):
        print(f"        its rank among the candidates: 1st {share(int((best_rank == 1).sum()), len(best_rank))}, "
              f"2nd-5th {share(int(best_rank.between(2, 5).sum()), len(best_rank))}, "
              f"below 5th {share(int((best_rank > 5).sum()), len(best_rank))}")
    print(f"    ... placed by the ranker: {share(found_placed.incident_number.nunique(), len(coded))}")
    print(f"    ... placed inside the coded district: {share(found_placed[found_placed.in_district].incident_number.nunique(), len(coded))}")
    print(f"    with a district filter (best candidate inside the coded district has the coded name): {share(len(filtered), len(coded))}")
    print(f"    found names that carried a word such as 'village' or 'police station': "
          f"{share(int(found.search_name.str.contains(GENERIC, case=False, regex=True).sum()), len(found))}")
    print("  feature codes of placed places below the district:")
    print(below.feature_code.value_counts().head(8).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "report"])
    ap.add_argument("split", choices=["validation", "test"])
    ap.add_argument("--limit", type=int, help="only the first N incidents in labeling order")
    ap.add_argument("--strip", action="store_true", help="drop words such as village or police station before lookup")
    ap.add_argument("--top-k", type=int, default=TOP_K, help="candidates kept per place")
    ap.add_argument("--fuzzy", type=int, default=0, help="Elasticsearch fuzziness for every GeoNames lookup (1 or 2)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    run(args.split, args.limit, args.strip, args.top_k, args.fuzzy) if args.step == "run" \
        else report(args.split, args.strip, args.top_k, args.fuzzy)
