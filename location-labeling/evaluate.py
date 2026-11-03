"""Each step of the pipeline, and the other systems, scored against the hand labels (labels/labels.csv).

Preliminary: the validation labels are the ones the pipeline was tuned on, so these scores give a first look at where
each step works and fails, not a final result.

Steps 1 and 2 come from suggest.py's cached model answers (run it first; this script makes no model calls), and
steps 1 and 3 are rerun here with the current code.

1. Lookup: is the labeled record among the candidates for step 1's places?
2. Where: does step 2 name the labeled place (its candidates include the record, or the name is spelled alike),
   and with the labeled relation?
3. Resolve: given step 2's place, does the code pick the labeled record?

End to end, each system is scored by its distance from the label's map point: our pick, Mordecai 3.5 (its first
place below the district in the text, and its closest place as a ceiling), last year's Flan-T5-large through our
lookup (compare_pipelines.py), and the India team's pins. Our misses are split by the first step that failed.

Run from the repository root:  python location-labeling/evaluate.py
"""

import json
import statistics
from collections import Counter

import pandas as pd

from compare_pipelines import MORDECAI, same, slots
from suggest import (CACHE, DATA, ROOT, STATION_WORDS, Areas, candidates, extract, km, load_descriptors, norm,
                     read_jsonl, resolve, where)

SPLIT = "validation"
REL = {"in a village or town": "in", "near a village or town": "near", "within police station limits": "limits",
       "in a forest or area": "area", "subdistrict": "subdistrict", "district": "district"}
WHERE_REL = {"in": "in", "near": "near", "within police station limits": "limits", "in a forest or area": "area",
             "subdistrict": "subdistrict", "district": "district"}
ADMIN = ("ADM1", "ADM2", "PCL")  # Mordecai places at the state, district or country level
WITHIN = (1, 5, 25)  # km


class Cached:
    """Stands in for the OpenAI client: every answer must already be in suggest.py's cache."""

    def __getattr__(self, name):
        raise SystemExit("A model answer is not cached. Run suggest.py on this split first.")


def right(o, l):
    """Is option o the labeled record (a subdistrict by its ID, an OpenStreetMap area by its ID, else the SHRUG place)?"""
    if o is None:
        return False
    if l.precision == "subdistrict":
        return o["relation"] == "subdistrict" and o["subdistrict_id"] == l.subdistrict_id
    if l.area_osm_id:
        return o.get("osm_id") == l.area_osm_id
    return bool(l.shrid2) and bool(o.get("shrid2")) and o["shrid2"] in l.shrid2.split("|")


def point(o):
    if o is None:
        return None
    lat, lon = o.get("point_lat") or o.get("lat"), o.get("point_lon") or o.get("lon")
    return (float(lat), float(lon)) if lat not in (None, "") and lon not in (None, "") else None


def away(p, gold):
    return km(*p, *gold) if p and gold else None


def distances(values, n):
    """Median km over the incidents given a point, and how many of all n fall within each WITHIN."""
    got = [d for d in values if pd.notna(d)]
    row = {"with a point": f"{len(got)} of {n}", "median km": f"{statistics.median(got):.1f}" if got else ""}
    for k in WITHIN:
        row[f"within {k} km"] = sum(d <= k for d in got)
    return row


def main():
    labels = pd.read_csv(ROOT / "labels" / "labels.csv", dtype=str).fillna("")
    labels = labels[(labels.split == SPLIT) & (labels.precision != "skipped") & (labels.skip_reason == "")]
    sample = pd.read_csv(DATA / "sample.csv", dtype=str).fillna("").set_index("incident_number", drop=False)
    labels = labels.assign(o=lambda d: sample.loc[d.incident_number, "order"].astype(int).values).sort_values("o")  # the app's order now

    links = pd.read_csv(DATA / "satp_links.csv", dtype=str).dropna(subset=["entry_id"]).fillna("")
    entries = pd.read_csv(DATA / "satp_entries.csv", dtype=str)
    text = dict(zip(zip(entries["month"], entries["entry_id"]), entries["text"]))
    satp = {r.incident_number: [text[(r.entry_month, r.entry_id)]] + [text[tuple(a.split())] for a in r.also.split(";") if a]
            for r in links.itertuples()}
    places = pd.read_parquet(DATA / "places.parquet")
    osm = pd.read_parquet(DATA / "osm_places.parquet")
    areas = Areas()
    load_descriptors(places, osm)
    ext_cache, where_cache = (read_jsonl(CACHE / f) for f in ("extractions.jsonl", "where.jsonl"))
    t5 = {r["incident_number"]: r["output"] for r in map(json.loads, (DATA / "t5" / f"{SPLIT}.jsonl").open())}
    mordecai = {r["incident_number"]: r for r in map(json.loads, open(str(MORDECAI).format(split=SPLIT)))}

    rows = []
    for l in labels.itertuples():
        i = l.incident_number
        inc = sample.loc[i].to_dict()
        areas.tree(inc["state_id"])
        texts = satp.get(i, [])
        gold = (float(l.point_lat), float(l.point_lon)) if l.point_lat else None
        rel = REL[l.precision]

        # ours, step by step
        mentions = extract(Cached(), ext_cache, inc, texts)
        step1, _ = candidates(inc, [dict(m) for m in mentions], places, osm, areas)
        w = where(Cached(), where_cache, inc, texts)
        options, _, ranked, _ = resolve(inc, [dict(m) for m in mentions], w, places, osm, areas)
        pick = options[0] if ranked == 1 and options else None
        station = STATION_WORDS.sub(" ", w["police_station"]).strip(" .,")
        said = [n for n in (w["place"], w["other_end"], station) if n.strip()]
        w_rel = WHERE_REL[w["relation"]]
        if rel == "district":
            s2_place = w_rel == "district"
            ours_place = pick is None
        else:
            named = said if rel == "limits" else said[:len([n for n in (w["place"], w["other_end"]) if n.strip()])]  # the station field counts for limits only
            s2_place = any(right(o, l) and norm(o["query"]) in {norm(n) for n in named} for o in options) \
                or any(same(n, l.place_name or l.unit_name) for n in named)
            ours_place = right(pick, l)
        ours_both = ours_place and (rel == "district" or pick["relation"] == rel)
        found = rel == "district" or any(right(o, l) for o in step1 + options)

        if ours_both:
            failed = ""
        elif rel == "district":
            failed = "2: named a place" if w_rel != "district" else "3: picked a place"
        elif not found:
            failed = "1: record not found"
        elif not s2_place:
            failed = "2: another place"
        elif rel == "subdistrict" and pick and pick["subdistrict_id"] == l.subdistrict_id:  # a place inside it: too precise
            failed = "2: relation" if w_rel != rel else "3: relation"
        elif not ours_place:
            failed = "3: another record"
        else:
            failed = "2: relation" if w_rel != rel else "3: relation"

        # T5 through our lookup: its village as where it happened, its other locations as near
        s = slots(t5.get(i, ""))
        t5_mentions = [{"name": n, "kind": "village", "role": "where it happened", "quote": n} for n in s.get("village", [])] \
            + [{"name": n, "kind": "other", "role": "near where it happened", "quote": n} for n in s.get("other_locations", [])]
        t5_opts = candidates(inc, t5_mentions, places, osm, areas)[0] if t5_mentions else []
        t5_pick = next((o for o in t5_opts if o["relation"] != "subdistrict"), None)

        # Mordecai: its first place below the district in the text, and (as a ceiling) its closest place
        m_places = sorted((p for p in mordecai.get(i, {}).get("places", [])
                           if not p["no_match"] and not p["feature_code"].startswith(ADMIN)), key=lambda p: p["start_char"])
        m_first = (m_places[0]["lat"], m_places[0]["lon"]) if m_places else None
        m_best = min((away((p["lat"], p["lon"]), gold) for p in m_places), default=None) if gold else None

        pin = (float(sample.at[i, "latitude"]), float(sample.at[i, "longitude"])) if sample.at[i, "latitude"] else None

        rows.append({
            "order": l.o, "incident_number": i, "label": f"{rel} {l.place_name or l.unit_name}", "certainty": l.certainty,
            "step1_found": rel == "district" or any(right(o, l) for o in step1),
            "step2_place": s2_place, "step2_relation": w_rel == rel, "step2_answer": f"{w_rel} {', '.join(said)}".strip(),
            "step3_right": ours_place if s2_place else None,
            "ours_place": ours_place, "ours_both": ours_both, "failed": failed,
            "ours_relation": pick is None if rel == "district" else pick is not None and pick["relation"] == rel,
            "ours_pick": f"{pick['relation']} {pick['name']}" if pick else "none",
            "ours_km": away(point(pick), gold), "t5_place": right(t5_pick, l) if rel != "district" else t5_pick is None,
            "t5_km": away(point(t5_pick), gold), "mordecai_first_km": away(m_first, gold), "mordecai_best_km": m_best,
            "india_pin_km": away(pin, gold), "has_point": gold is not None,
        })

    r = pd.DataFrame(rows)
    out = DATA / f"evaluation_{SPLIT}.csv"
    r.to_csv(out, index=False)
    n, below = len(r), r[~r.label.str.startswith("district")]
    pd.set_option("display.width", 250)

    print(f"\n{SPLIT}: {n} labels ({len(below)} below the district, {r.has_point.sum()} with a map point). Preliminary.\n")
    print("By step (ours)")
    s3 = r[r.step2_place]
    print(pd.DataFrame([
        ("1. Labeled record among step 1's candidates", r.step1_found.sum(), n),
        ("2. Step 2 names the labeled place", r.step2_place.sum(), n),
        ("2. ... with the labeled relation", r.step2_relation.sum(), n),
        ("3. Right record, when step 2 named the right place", int(s3.step3_right.sum()), len(s3)),
        ("End to end: place", r.ours_place.sum(), n),
        ("End to end: place and relation", r.ours_both.sum(), n),
    ], columns=["", "right", "of"]).to_string(index=False))

    print("\nWhat step 3 does to step 2's answer")
    effect = []
    for what, before, after in (("place", "step2_place", "ours_place"), ("relation", "step2_relation", "ours_relation")):
        b, a = r[before].astype(bool), r[after].astype(bool)
        for name, mask in (("kept right", b & a), ("fixed", ~b & a), ("lost", b & ~a), ("wrong both times", ~b & ~a)):
            effect.append((what, name, int(mask.sum()), ", ".join(r.label[mask].str.split(n=1).str[1]) if name in ("fixed", "lost") else ""))
    print(pd.DataFrame(effect, columns=["", "step 3", "labels", "which"]).to_string(index=False))

    print("\nWhere our misses come from (first step that failed)")
    print(pd.Series(Counter(r.failed[r.failed != ""])).sort_index().to_string())

    print("\nStep 2 relation: label (rows) by step 2's answer (columns)")
    print(pd.crosstab(r.label.str.split().str[0], pd.Series([x.split()[0] for x in r.step2_answer], name="step 2")).to_string())

    pts = r[r.has_point]
    print(f"\nDistance from the label's map point ({len(pts)} labels with a point)")
    print(pd.DataFrame([
        {"system": "Ours", **distances(pts.ours_km, len(pts))},
        {"system": "Flan-T5-large + our lookup", **distances(pts.t5_km, len(pts))},
        {"system": "Mordecai 3.5, first place", **distances(pts.mordecai_first_km, len(pts))},
        {"system": "Mordecai 3.5, closest place (ceiling)", **distances(pts.mordecai_best_km, len(pts))},
        {"system": "India team pin", **distances(pts.india_pin_km, len(pts))},
    ]).to_string(index=False))

    print("\nPlace right, by certainty of the label")
    print(r.groupby("certainty").agg(labels=("ours_place", "size"), ours=("ours_place", "sum"), t5=("t5_place", "sum")).to_string())

    print("\nMisses")
    print(r[r.failed != ""][["order", "label", "step2_answer", "ours_pick", "failed"]].to_string(index=False))
    print(f"\nPer incident: {out}")


if __name__ == "__main__":
    main()
