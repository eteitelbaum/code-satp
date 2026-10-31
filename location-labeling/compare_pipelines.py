"""Mordecai, last year's T5 pipeline, and ours, side by side on the same incidents (GitHub #22).

Three steps, each scored against the coders' village (a rough reference, not gold labels), with the
same name test as mordecai_baseline.py (spelling >= 0.85, after dropping words such as "village"):

1. Names found: is the coded village among the place names the system finds, whatever their role?
2. Event village: does the system give it as where the event happened? Mordecai has no such step.
3. Placed: is the system's answer the coded village, at a point (or subdistrict) inside the coded district?

The systems:
- Mordecai 3.5 in its best configuration (generic words stripped, fuzzy search 1, 100 candidates):
  run mordecai_baseline.py first.
- Last year's Flan-T5-large (models/location-models/location_extraction_seq2seq.ipynb), run here on
  the coded summaries, its training input. In place of the old Google geocoding step, its names go
  through our candidate search (suggest.candidates): the village as where it happened, other
  locations as near it. That isolates the extractor.
- Our pipeline: suggest.py's GPT-4o-mini extraction, candidate search, and pick
  (data/mentions.csv, data/suggestions.csv).

Mordecai and our pipeline read the coded summary and the SATP entry; T5 reads the summary only, so
step 1 is also given from the summary alone.

The T5 checkpoint goes in model-checkpoints/flan-t5-large (gitignored; from the Colab results folder
satp-results/location-extraction/flan-t5-large_finetuned_model). Its outputs are cached in
data/t5/<split>.jsonl, one line per incident.

Run from the repository root:  python location-labeling/compare_pipelines.py validation
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import shapely

from suggest import GENERIC, SETTLEMENT_KINDS, Areas, candidates, norm, spelling

ROOT = Path(__file__).parent
DATA = ROOT / "data"
T5_MODEL = ROOT.parent / "model-checkpoints" / "flan-t5-large"
T5_OUT = DATA / "t5"
MORDECAI = DATA / "mordecai" / "{split}_stripped_k100_fuzzy1.jsonl"
SAME_NAME = 0.85
T5_BATCH = 8
SLOTS = re.compile(r"\b(state|district|village|other_locations):\s*")


def bare(name):
    return re.sub(r"\s+", " ", re.sub(GENERIC, " ", name, flags=re.I)).strip()


def same(name, village):
    a, b = norm(bare(name)), norm(bare(village))
    return bool(a and b) and spelling(a, b) >= SAME_NAME


def in_summary(name, summary):
    return norm(bare(name)) in norm(summary)


def run_t5(split, sample):
    """Last year's Flan-T5-large on the coded summaries, with its training prompt; cached."""
    path = T5_OUT / f"{split}.jsonl"
    done = {}
    if path.exists():
        done = {r["incident_number"]: r["output"] for r in map(json.loads, path.open())}
    todo = sample[~sample.incident_number.isin(done)]
    if len(todo):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        sys.path.insert(0, str(ROOT.parent / "models" / "location-models"))
        from utils.data_utils import prepare_location_seq2seq_data

        print(f"T5: {len(todo)} incidents to run ({len(done)} already done)")
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(T5_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL).to(device).eval()
        prompts = prepare_location_seq2seq_data(todo.assign(human_annotated_location=""))["input"]
        T5_OUT.mkdir(exist_ok=True)
        with path.open("a") as f:
            for i in range(0, len(todo), T5_BATCH):
                batch = tok(prompts[i:i + T5_BATCH], max_length=512, truncation=True, padding=True,
                            return_tensors="pt").to(device)
                with torch.no_grad():  # the notebook's generation settings
                    out = model.generate(**batch, max_length=128, num_beams=4)
                for iid, text in zip(todo.incident_number.iloc[i:i + T5_BATCH], tok.batch_decode(out, skip_special_tokens=True)):
                    done[iid] = text
                    f.write(json.dumps({"incident_number": iid, "output": text}) + "\n")
                if (i // T5_BATCH) % 10 == 0:
                    print(f"  {min(i + T5_BATCH, len(todo))} of {len(todo)}", flush=True)
    return done


def slots(output):
    """'state: X, district: Y, village: Z, other_locations: A, B' -> {'village': ['Z'], 'other_locations': ['A', 'B'], ...}"""
    parts = SLOTS.split(output)
    return {label: [v.strip() for v in value.strip().strip(",").split(",") if v.strip()]
            for label, value in zip(parts[1::2], parts[2::2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["validation", "test"])
    args = ap.parse_args()

    sample = pd.read_csv(DATA / "sample.csv", dtype=str).fillna("")
    sample = sample[sample.split == args.split].assign(o=lambda d: d["order"].astype(int)).sort_values("o")
    inc = sample.set_index("incident_number")
    coded = inc[inc.village_name.str.strip() != ""]

    subs = pd.read_parquet(DATA / "subdistrict_shapes.parquet")
    tree = shapely.STRtree(shapely.from_wkb(subs["wkb"].values))
    sub_keys = set(zip(subs.state_id.astype(str), subs.district_id.astype(str), subs.subdistrict_id.astype(str)))

    def district_of(lon, lat):
        pi, si = tree.query(shapely.points(lon, lat), predicate="intersects")
        hit = pd.Series(subs["district_id"].astype(str).values[si], index=pi).groupby(level=0).first()
        return hit.reindex(range(len(lon))).fillna("").values

    def options_in_district(opts):
        """For suggest.py's options: does each lie in its incident's coded district (subdistrict options by their subdistrict)?"""
        opts = opts.copy()
        point = opts.lat.astype(str).ne("") & opts.lat.notna()
        opts["in_d"] = False
        if point.any():
            opts.loc[point, "in_d"] = district_of(opts.loc[point, "lon"].astype(float).values,
                                                  opts.loc[point, "lat"].astype(float).values) \
                == inc.loc[opts.loc[point, "incident_number"], "district_id_2011"].values
        nopt = ~point & opts.subdistrict_id.astype(str).ne("")
        opts.loc[nopt, "in_d"] = [(inc.at[i, "state_id"], inc.at[i, "district_id_2011"], str(s)) in sub_keys
                                  for i, s in zip(opts.loc[nopt, "incident_number"], opts.loc[nopt, "subdistrict_id"])]
        opts["right"] = opts.in_d & [same(n, coded.at[i, "village_name"]) if i in coded.index else False
                                     for i, n in zip(opts.incident_number, opts.name.astype(str))]
        return opts

    rows = {i: {} for i in coded.index}

    # ---- Mordecai
    mordecai = [json.loads(line) for line in open(str(MORDECAI).format(split=args.split))]
    for r in mordecai:
        i = r["incident_number"]
        if i not in rows:
            continue
        village, summary_chars = coded.at[i, "village_name"], r["summary_chars"]
        found = [p for p in r["places"] if same(p["search_name"], village)]
        rows[i]["mordecai_found"] = bool(found)
        rows[i]["mordecai_found_summary"] = any(p["start_char"] < summary_chars for p in found)
        rows[i]["_mordecai"] = found
    found = [(i, n, p) for i, row in rows.items() for n, p in enumerate(row.pop("_mordecai", []))]
    placed = [(i, p) for i, _, p in found if not p["no_match"]]
    if placed:
        ok = district_of([p["lon"] for _, p in placed], [p["lat"] for _, p in placed]) \
            == inc.loc[[i for i, _ in placed], "district_id_2011"].values
        for (i, p), d in zip(placed, ok):
            rows[i]["mordecai_placed"] = rows[i].get("mordecai_placed", False) or bool(d and same(p["name"], coded.at[i, "village_name"]))
    cands = pd.DataFrame([{"i": i, "n": n, **c} for i, n, p in found for c in p.get("candidates") or []])
    if len(cands):
        cands["in_d"] = district_of(cands.lon.astype(float).values, cands.lat.astype(float).values) \
            == inc.loc[cands.i, "district_id_2011"].values
        cands["named"] = [same(c, coded.at[i, "village_name"]) for i, c in zip(cands.i, cands.name)]
        for i in cands[cands.in_d & cands.named].i.unique():
            rows[i]["mordecai_retrieved"] = True
        # a district filter: each name's best-scoring candidate inside the coded district
        best = cands[cands.in_d].sort_values("score", ascending=False).drop_duplicates(["i", "n"])
        for i in best[best.named].i.unique():
            rows[i]["mordecai_filtered"] = True

    # ---- T5, then our candidate search on its names
    t5 = run_t5(args.split, sample)
    places = pd.read_parquet(DATA / "places.parquet")
    osm = pd.read_parquet(DATA / "osm_places.parquet")
    areas = Areas()
    t5_options, t5_event = [], {}
    for i, out in t5.items():
        s = slots(out)
        village, others = s.get("village", []), s.get("other_locations", [])
        t5_event[i] = village
        if i in rows:
            rows[i]["t5_found"] = any(same(n, coded.at[i, "village_name"]) for n in village + others)
            rows[i]["t5_event"] = any(same(n, coded.at[i, "village_name"]) for n in village)
        mentions = [{"name": n, "kind": "village", "role": "where it happened", "quote": n} for n in village] \
            + [{"name": n, "kind": "other", "role": "near where it happened", "quote": n} for n in others]
        if i in coded.index and mentions:
            areas.tree(inc.at[i, "state_id"])
            opts, _ = candidates({"incident_number": i, **inc.loc[i].to_dict()}, mentions, places, osm, areas)
            t5_options += [{"incident_number": i, "rank": k, **o} for k, o in enumerate(opts, 1)]
    t5_opts = options_in_district(pd.DataFrame(t5_options).fillna({"lat": "", "lon": "", "subdistrict_id": ""}))
    for i, g in t5_opts.groupby("incident_number"):
        rows[i]["t5_retrieved"] = bool(g.right.any())
        rows[i]["t5_placed"] = bool(g[g["rank"] == 1].right.any())

    # ---- ours
    mentions = pd.read_csv(DATA / "mentions.csv", dtype=str).fillna("")
    for i, g in mentions[mentions.incident_number.isin(coded.index)].groupby("incident_number"):
        village, summary = coded.at[i, "village_name"], coded.at[i, "incident_summary"]
        hit = g[[same(n, village) for n in g.name]]
        rows[i]["ours_found"] = len(hit) > 0
        rows[i]["ours_found_summary"] = any(in_summary(n, summary) for n in hit.name)
        rows[i]["ours_event"] = bool(((hit.role == "where it happened") & hit.kind.isin(SETTLEMENT_KINDS)).any())
    sug = pd.read_csv(DATA / "suggestions.csv", dtype=str).fillna("")
    sug = options_in_district(sug[sug.incident_number.isin(coded.index) & (sug["rank"] != "0")].assign(rank=lambda d: d["rank"].astype(int)))
    for i, g in sug.groupby("incident_number"):
        rows[i]["ours_retrieved"] = bool(g.right.any())
        rows[i]["ours_placed"] = bool(g[g["rank"] == 1].right.any())
        rows[i]["ours_pick"] = bool(g[g.model_pick == "True"].right.any())

    res = pd.DataFrame.from_dict(rows, orient="index").fillna(False).astype(bool)
    res.index.name = "incident_number"
    res.to_csv(DATA / f"compare_{args.split}.csv")

    # event-village precision over all incidents: of those where the system names an event village, how often is it the coded one?
    ev = mentions[(mentions.role == "where it happened") & mentions.kind.isin(SETTLEMENT_KINDS)]
    ours_named = set(ev.incident_number) & set(inc.index)
    t5_named = {i for i, v in t5_event.items() if v and i in inc.index}

    n = len(coded)

    def pct(col):
        return f"{res[col].sum() / n:.0%}" if col in res else "n/a"

    def precision(named, col):
        return f"{res.loc[res.index.isin(named), col].sum() / len(named):.0%}" if named else "n/a"

    print(f"\n{args.split}: {len(inc)} incidents, {n} with a coded village (the reference for every row)\n")
    table = [
        ("1. Names found, any role", pct("mordecai_found"), pct("t5_found"), pct("ours_found")),
        ("   ... from the coded summary alone", pct("mordecai_found_summary"), pct("t5_found"), pct("ours_found_summary")),
        ("2. Event village (recall)", "n/a", pct("t5_event"), pct("ours_event")),
        ("   precision, of incidents naming one", "n/a", precision(t5_named, "t5_event"), precision(ours_named, "ours_event")),
        ("3. Among the options or candidates", pct("mordecai_retrieved"), pct("t5_retrieved"), pct("ours_retrieved")),
        ("   placed: the system's answer", pct("mordecai_placed"), pct("t5_placed"), pct("ours_placed")),
        ("   placed: with a district filter / GPT-4o-mini's pick", pct("mordecai_filtered"), "", pct("ours_pick")),
    ]
    print(pd.DataFrame(table, columns=["", "Mordecai 3.5", "T5 + our search", "Ours"]).to_string(index=False))
    print(f"\nPer incident: {DATA / f'compare_{args.split}.csv'}")


if __name__ == "__main__":
    main()
