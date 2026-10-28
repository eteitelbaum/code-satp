"""Precompute location suggestions for the labeling app (GitHub #9).

For each incident in a split:
1. Extraction: GPT-4o-mini reads the coded summary and the full SATP entry and lists the
   places named, with each place's role (where it happened, near it, a jurisdiction, ...).
2. Candidates: each place is matched to SHRUG villages, towns, and subdistricts in the
   coded 2011 district (or, failing a close match there, elsewhere in the state, which
   catches events the coders placed in a neighboring district). Places SHRUG does not list
   are looked up in OpenStreetMap
   (Nominatim, one request a second, limited to the district's area), and a hit is tied to
   the village or subdistrict it falls in. Options are ranked by spelling and context
   (the place's role, whether it lies in a named subdistrict, distance to a named police
   station); the weights are a first guess, to be tuned on the validation labels.
3. Selection: GPT-4o-mini sees the text and the top five options and picks one (or none),
   with a one-sentence reason.

Model calls and OpenStreetMap lookups are cached in location-labeling/data/suggest/, so a
rerun only does new work. The app reads the outputs and calls nothing. Each run appends its
tokens, estimated cost, and any failed incidents to data/suggest/runs.jsonl (shown in the
app's sidebar). Temporary API errors are retried; if the account is out of credit the run
stops, keeping what it has done, and a rerun picks up from there.

Run from the repository root after link_satp.py, with OPENAI_API_KEY in .env:
    python location-labeling/suggest.py validation [--limit N]

Output (location-labeling/data/):
    suggestions.csv  one row per incident and option: rank, source, place, evidence, model pick and reason
    mentions.csv     one row per extracted place, with what it matched
"""

import argparse
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
import shapely
from dotenv import load_dotenv
import openai
from openai import OpenAI
from pydantic import BaseModel
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CACHE = DATA / "suggest"
MODEL = "gpt-4o-mini"
TOP = 5
MATCH = 0.80  # spelling similarity for a SHRUG candidate
SURE = 0.92  # a SHRUG match this close needs no OpenStreetMap lookup
NOMINATIM = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "code-satp research (https://github.com/eteitelbaum/code-satp)"}
PRICE = {"prompt": 0.15 / 1e6, "completion": 0.60 / 1e6}  # GPT-4o-mini, US$ per token (estimate)
BILLING = "https://platform.openai.com/settings/organization/billing/overview"
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

# ranking weights, added to spelling similarity
ROLE_WEIGHT = {"where it happened": 0.3, "near where it happened": 0.2, "jurisdiction": -0.1}
IN_NAMED_SUBDISTRICT = 0.2
NEAR_POLICE_STATION = 0.1  # within 15 km of a named police station
SETTLEMENT_KINDS = {"village", "town", "hamlet", "locality", "police station", "camp", "other"}
OTHER_DISTRICT = -0.1  # a match outside the coded district, which needs a closer spelling (SURE)
NEAR_SPELLING = 0.05  # keep a name's SHRUG matches within this of its best

# words that describe a place rather than name it
GENERIC = (r"\b(villages?|areas?|reserve forests?|forests?|jungles?|hills?|hillocks?|valley|police stations?|ps|outposts?|camps?|blocks?|district|mandal|tehsil|taluka|panchayat|sub-?division|road|upgraded|middle|primary|high|schools?|the)\b")


# ---------------------------------------------------------------- helpers

def norm(x):
    return re.sub(r"[^a-z]", "", str(x).lower())


def spelling(a, b, **kwargs):
    """The app's spelling similarity: average of Jaro-Winkler and an edit-distance ratio."""
    return (JaroWinkler.normalized_similarity(a, b) + fuzz.ratio(a, b) / 100) / 2


def query_names(name):
    """The name without generic words, and its parts when it joins several ("Barigada-Sashank")."""
    clean = re.sub(r"\s+", " ", re.sub(GENERIC, " ", name, flags=re.I)).strip(" .,-")
    parts = [x.strip() for x in re.split(r"[-/&,]| and ", clean) if len(norm(x)) >= 3]
    return list(dict.fromkeys([clean] + (parts if len(parts) > 1 else [])))


def km(lat1, lon1, lat2, lon2):
    return 111.32 * math.hypot(lat1 - lat2, (lon1 - lon2) * math.cos(math.radians(lat1)))


def call_model(client, **kwargs):
    """One model call, retried on temporary errors; stops the run if the account is out of credit."""
    for wait in (5, 20, 60, None):
        try:
            r = client.chat.completions.parse(**kwargs)
            USAGE["calls"] += 1
            USAGE["prompt_tokens"] += r.usage.prompt_tokens
            USAGE["completion_tokens"] += r.usage.completion_tokens
            return r
        except openai.RateLimitError as e:
            if "insufficient_quota" in str(e):
                raise SystemExit(f"OpenAI credit exhausted: refill at {BILLING}, then rerun (finished work is cached).")
            error = e
        except openai.AuthenticationError:
            raise SystemExit("OpenAI rejected the API key: check OPENAI_API_KEY in .env.")
        except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as e:
            error = e
        if wait is None:
            raise error
        time.sleep(wait)


def read_jsonl(path):
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["key"]: r for r in rows}


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------- 1. extraction

class Mention(BaseModel):
    name: str
    kind: Literal["village", "town", "hamlet", "locality", "police station", "camp", "forest", "hill", "river",
                  "road", "mandal or block", "district", "state", "other"]
    role: Literal["where it happened", "near where it happened", "jurisdiction", "other place in this incident",
                  "another incident"]
    quote: str


class Extraction(BaseModel):
    mentions: list[Mention]


EXTRACT_PROMPT = """You help geocode incidents in India's Maoist insurgency.

The CODED SUMMARY describes one incident. The FULL SATP ENTRY, when given, is the news report it came from: it may add detail about the same incident, and it may also describe other incidents or follow-ups.

List every named place below the state level that the texts mention, once each, spelled as written. Give the proper name only, without words such as village, police station, forest, hill, camp, school, or district ("Kurdeg", not "Kurdeg Police Station"). For each place give:
- kind: what the place is (village, town, hamlet, locality, police station, camp, forest, hill, river, road, mandal or block, district, state, other). A mandal, block, tehsil, taluka, or sub-division is "mandal or block".
- role, relative to the incident in the CODED SUMMARY:
  - "where it happened": the event took place in or at this place
  - "near where it happened": the event took place near it, or in a forest, hill, or area named after or tied to it
  - "jurisdiction": an administrative or police area containing the spot, e.g. "under X Police Station limits", "X mandal", "X district"
  - "other place in this incident": named in the incident but not where it happened (where victims lived, where forces came from, where bodies were taken)
  - "another incident": belongs only to a different incident in the full entry
- quote: the shortest phrase from the text that shows the role."""


def extract(client, cache, inc, texts):
    key = inc["incident_number"]
    if key in cache:
        return cache[key]["mentions"]
    user = f"CODED SUMMARY:\n{inc['incident_summary']}"
    if texts:
        user += "\n\nFULL SATP ENTRY:\n" + "\n\n".join(texts)
    r = call_model(
        client, model=MODEL, temperature=0, response_format=Extraction,
        messages=[{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": user}],
    )
    mentions = [m.model_dump() for m in r.choices[0].message.parsed.mentions]
    row = {"key": key, "model": MODEL, "mentions": mentions}
    append_jsonl(CACHE / "extractions.jsonl", row)
    cache[key] = row
    return mentions


# ---------------------------------------------------------------- 2. candidates

def osm_lookup(cache, name, district_key, viewbox):
    """Nominatim places named `name` inside the district's bounding box (cached)."""
    key = f"{norm(name)}|{district_key}"
    if key in cache:
        return cache[key]["hits"]
    time.sleep(1.1)  # Nominatim's usage policy: at most one request a second
    params = {"q": name, "format": "jsonv2", "countrycodes": "in", "viewbox": viewbox, "bounded": 1, "limit": 3}
    try:
        r = requests.get(NOMINATIM, params=params, headers=HEADERS, timeout=30)
    except requests.RequestException:
        return []  # not cached, so a rerun tries again
    hits = [{"name": h.get("name") or h["display_name"].split(",")[0], "type": h.get("addresstype") or h.get("type"),
             "lat": float(h["lat"]), "lon": float(h["lon"]), "display": h["display_name"]}
            for h in (r.json() if r.ok else [])]
    row = {"key": key, "hits": hits}
    append_jsonl(CACHE / "osm.jsonl", row)
    cache[key] = row
    return hits


class Shapes:
    """Which SHRUG village or subdistrict a point falls in."""

    def __init__(self):
        self.places = pd.read_parquet(DATA / "place_shapes.parquet")
        self.subs = pd.read_parquet(DATA / "subdistrict_shapes.parquet")
        self._trees = {}

    def _tree(self, kind, state_id):
        if (kind, state_id) not in self._trees:
            df = (self.places if kind == "place" else self.subs)
            df = df[df.state_id == state_id].reset_index(drop=True)
            self._trees[(kind, state_id)] = (df, shapely.STRtree(shapely.from_wkb(df["wkb"].values)))
        return self._trees[(kind, state_id)]

    def at(self, state_id, lat, lon):
        point = shapely.Point(lon, lat)
        df, tree = self._tree("place", state_id)
        hit = tree.query(point, predicate="intersects")
        if len(hit):
            return df.loc[hit[0], "shrid2"], None
        df, tree = self._tree("sub", state_id)
        hit = tree.query(point, predicate="intersects")
        return None, (df.loc[hit[0], "subdistrict_id"] if len(hit) else None)


def candidates(inc, mentions, places, shapes, osm_cache, use_osm=True):
    """Options for one incident, best first, and the mentions with what each matched."""
    district = places[(places.state_id == inc["state_id"]) & (places.district_id == inc["district_id_2011"])]
    if district.empty:
        return [], mentions
    keys = district["name"].map(norm)
    subs = district.drop_duplicates("subdistrict_id")[["subdistrict_id", "subdistrict_name"]]
    district_names = {norm(inc["district"]), norm(inc["district_name_2011"])}
    pad = 0.05
    viewbox = f"{district.lon.min() - pad},{district.lat.max() + pad},{district.lon.max() + pad},{district.lat.min() - pad}"

    # context: subdistricts named in the text, and named police stations' locations
    named_subs, stations = set(), []
    for m in mentions:
        if m["role"] == "another incident":
            continue
        if m["kind"] == "mandal or block":
            for _, s in subs.iterrows():
                if spelling(norm(m["name"]), norm(s["subdistrict_name"])) >= 0.85:
                    named_subs.add(s["subdistrict_id"])
        if m["kind"] == "police station":
            exact = district[keys == norm(m["name"])]
            if len(exact):
                stations.append((exact.iloc[0]["lat"], exact.iloc[0]["lon"]))

    options = []
    for m in mentions:
        m["matched"] = ""
        if m["role"] in ("another incident", "other place in this incident") or m["kind"] in ("district", "state") \
                or (norm(m["name"]) in district_names and m["kind"] not in SETTLEMENT_KINDS):  # keep "Ranchi" the town
            continue
        if m["role"] == "jurisdiction" and m["kind"] != "mandal or block":
            # "under X Police Station": the subdistrict X lies in, not X itself (protocol rule 2)
            for q in query_names(m["name"]):
                for _, sim, i in process.extract(norm(q), keys, scorer=spelling, limit=3):
                    if sim < 0.88:  # close enough to name the station's place ("Bandharia", "Bhandaria")
                        continue
                    p = district.loc[i]
                    options.append({"relation": "subdistrict", "source": "SHRUG",
                                    "name": str(p["subdistrict_name"]).title(), "shrid2": "",
                                    "subdistrict_id": p["subdistrict_id"], "lat": None, "lon": None,
                                    "score": sim + ROLE_WEIGHT["jurisdiction"], "query": m["name"],
                                    "evidence": f"subdistrict of {p['name']}, a jurisdiction in the text (“{m['quote']}”)"})
                    m["matched"] = f"SHRUG {sim:.2f} (jurisdiction)"
            continue
        weight = ROLE_WEIGHT.get(m["role"], 0.0)
        relation = "in" if m["role"] == "where it happened" and m["kind"] in SETTLEMENT_KINDS else "near"
        if m["kind"] == "mandal or block":
            for _, s in subs.iterrows():
                sim = spelling(norm(m["name"]), norm(s["subdistrict_name"]))
                if sim >= 0.85:
                    options.append({"relation": "subdistrict", "source": "SHRUG", "name": s["subdistrict_name"].title(),
                                    "shrid2": "", "subdistrict_id": s["subdistrict_id"], "lat": None, "lon": None,
                                    "score": sim + 0.1 * (weight + 0.1), "query": m["name"],
                                    "evidence": f"subdistrict named in the text (“{m['quote']}”), spelling {sim:.2f}"})
                    m["matched"] = "SHRUG subdistrict"
            continue
        names = query_names(m["name"])
        hits = {i: max(spelling(norm(q), keys[i]) for q in names)
                for q in names for _, _, i in process.extract(norm(q), keys, scorer=spelling, limit=8)}
        hits = {i: sim for i, sim in hits.items() if sim >= MATCH}
        pool = district
        if max(hits.values(), default=0) < SURE:  # try the rest of the state
            state = places[places.state_id == inc["state_id"]]
            skeys = state["name"].map(norm)
            far = {i: max(spelling(norm(q), skeys[i]) for q in names)
                   for q in names for _, _, i in process.extract(norm(q), skeys, scorer=spelling, limit=5)}
            far = {i: sim for i, sim in far.items() if sim >= SURE and i not in district.index}
            if far:
                hits.update(far)
                pool = places
        best = max(hits.values(), default=0.0)
        for i, sim in sorted(hits.items(), key=lambda x: -x[1]):
            if sim < best - NEAR_SPELLING:
                continue
            p = pool.loc[i]
            context = []
            score = sim + weight
            if p["district_id"] != inc["district_id_2011"]:
                score += OTHER_DISTRICT
                context.append(f"in {p['district_name'].title()} district, not the coded one")
            if p["subdistrict_id"] in named_subs:
                score += IN_NAMED_SUBDISTRICT
                context.append("in the named subdistrict")
            if stations and min(km(p["lat"], p["lon"], la, lo) for la, lo in stations) <= 15:
                score += NEAR_POLICE_STATION
                context.append("within 15 km of the named police station")
            options.append({"relation": relation, "source": "SHRUG", "name": p["name"], "shrid2": p["shrid2"],
                            "subdistrict_id": p["subdistrict_id"], "lat": p["lat"], "lon": p["lon"], "score": score,
                            "query": m["name"],
                            "evidence": " · ".join([f"SHRUG {p['place_type']} in {str(p['subdistrict_name']).title()} subdistrict, "
                                                    f"spelling {sim:.2f} to “{m['name']}”",
                                                    f"{m['role']} (“{m['quote']}”)"] + context)})
        if best:
            m["matched"] = f"SHRUG {best:.2f}"
        if best < SURE and use_osm and m["kind"] in SETTLEMENT_KINDS | {"forest", "hill"}:
            for h in osm_lookup(osm_cache, names[0], f"{inc['state_id']}-{inc['district_id_2011']}", viewbox):
                sim = spelling(norm(names[0]), norm(h["name"]))
                if sim < MATCH:
                    continue
                shrid, sub = shapes.at(inc["state_id"], h["lat"], h["lon"])
                inside = district[district.shrid2 == shrid]
                where = f"falls in {inside.iloc[0]['name']}" if len(inside) else "outside every village"
                score = sim + weight - 0.05  # a little below an equal SHRUG match
                if (inside.iloc[0]["subdistrict_id"] if len(inside) else sub) in named_subs:
                    score += IN_NAMED_SUBDISTRICT
                options.append({"relation": relation, "source": "OpenStreetMap", "name": h["name"], "shrid2": shrid or "",
                                "subdistrict_id": (inside.iloc[0]["subdistrict_id"] if len(inside) else sub) or "",
                                "lat": h["lat"], "lon": h["lon"], "score": score, "query": m["name"],
                                "evidence": f"OpenStreetMap {h['type']}, {where} · {m['role']} (“{m['quote']}”)"})
                m["matched"] += (" · " if m["matched"] else "") + "OpenStreetMap"

    # one option per place: keep the best-scoring
    seen, out = set(), []
    for o in sorted(options, key=lambda o: -o["score"]):
        k = (o["source"], o["shrid2"] or o["subdistrict_id"] if o["source"] == "SHRUG" else (round(o["lat"], 4), round(o["lon"], 4)))
        if k not in seen:
            seen.add(k)
            out.append(o)
    return out[:TOP], mentions


# ---------------------------------------------------------------- 3. selection

class Pick(BaseModel):
    option: int  # 1-5, or 0 for none of these
    reason: str


SELECT_PROMPT = """You help geocode incidents in India's Maoist insurgency. Given the incident text and numbered options for where it happened, choose the option that best gives where the event in the CODED SUMMARY took place, or 0 if none fits.

Rules: prefer the most specific place the text supports; a place the event happened in or at beats one it happened near; a forest, hill, or area named after a place counts as near that place; "under X Police Station" names a jurisdiction, not the place; ignore places that belong to other incidents in the full entry. The coders' district is sometimes wrong, so a close match in a neighboring district can be the right answer.

Give the option number and a one-sentence reason that cites the text."""


def select(client, cache, inc, texts, options):
    listing = "\n".join(f"{i}. {o['relation']} {o['name']} — {o['evidence']}" for i, o in enumerate(options, 1))
    key = inc["incident_number"] + "|" + hashlib.sha1(listing.encode()).hexdigest()[:12]
    if key in cache:
        return cache[key]["option"], cache[key]["reason"]
    user = f"CODED SUMMARY:\n{inc['incident_summary']}"
    if texts:
        user += "\n\nFULL SATP ENTRY:\n" + "\n\n".join(texts)
    user += f"\n\nOPTIONS:\n{listing}\n0. none of these"
    r = call_model(
        client, model=MODEL, temperature=0, response_format=Pick,
        messages=[{"role": "system", "content": SELECT_PROMPT}, {"role": "user", "content": user}],
    )
    pick = r.choices[0].message.parsed
    option = pick.option if 0 <= pick.option <= len(options) else 0
    row = {"key": key, "model": MODEL, "option": option, "reason": pick.reason}
    append_jsonl(CACHE / "selections.jsonl", row)
    cache[key] = row
    return option, pick.reason


# ---------------------------------------------------------------- run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["validation", "test"])
    ap.add_argument("--limit", type=int, help="only the first N incidents in labeling order")
    ap.add_argument("--no-osm", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    client = OpenAI()
    sample = pd.read_csv(DATA / "sample.csv", dtype=str).fillna("")
    sample = sample[sample.split == args.split].assign(o=lambda d: d["order"].astype(int)).sort_values("o")
    if args.limit:
        sample = sample.head(args.limit)
    links = pd.read_csv(DATA / "satp_links.csv", dtype=str).dropna(subset=["entry_id"]).fillna("")
    entries = pd.read_csv(DATA / "satp_entries.csv", dtype=str)
    text = dict(zip(zip(entries["month"], entries["entry_id"]), entries["text"]))
    satp = {r.incident_number: [text[(r.entry_month, r.entry_id)]] + [text[tuple(a.split())] for a in r.also.split(";") if a]
            for r in links.itertuples()}
    places = pd.read_parquet(DATA / "places.parquet")
    shapes = Shapes()
    ext_cache, sel_cache, osm_cache = (read_jsonl(CACHE / f) for f in ("extractions.jsonl", "selections.jsonl", "osm.jsonl"))

    rows, mention_rows, failed = [], [], []
    run = {"key": datetime.now().isoformat(timespec="seconds"), "split": args.split, "incidents": len(sample)}

    def log(status):
        cost = USAGE["prompt_tokens"] * PRICE["prompt"] + USAGE["completion_tokens"] * PRICE["completion"]
        append_jsonl(CACHE / "runs.jsonl", {**run, **USAGE, "estimated_cost_usd": round(cost, 4),
                                            "failed": failed, "status": status})
        print(f"{status}: {USAGE['calls']} model calls, {USAGE['prompt_tokens'] + USAGE['completion_tokens']:,} tokens, "
              f"about ${cost:.2f}; failed incidents: {len(failed)}")

    for n, inc in enumerate(sample.to_dict("records"), 1):
        texts = satp.get(inc["incident_number"], [])
        try:
            mentions = extract(client, ext_cache, inc, texts)
            options, mentions = candidates(inc, mentions, places, shapes, osm_cache, use_osm=not args.no_osm)
            pick, reason = select(client, sel_cache, inc, texts, options) if options else (0, "no options")
        except SystemExit as e:
            log(f"stopped at incident {n} of {len(sample)}: {e}")
            sys.exit(str(e))
        except Exception as e:  # one incident's failure should not end the run
            failed.append(inc["incident_number"])
            print(f"{inc['incident_number']} failed: {e}", flush=True)
            continue
        for m in mentions:
            mention_rows.append({"incident_number": inc["incident_number"], **m})
        for rank, o in enumerate(options, 1):
            rows.append({"incident_number": inc["incident_number"], "rank": rank, **o,
                         "model_pick": rank == pick, "model_reason": reason if rank == pick else ""})
        if not options or pick == 0:
            rows.append({"incident_number": inc["incident_number"], "rank": 0, "relation": "", "source": "",
                         "name": "", "model_pick": True, "model_reason": reason})
        if n % 25 == 0:
            print(f"{n} of {len(sample)}", flush=True)

    sug = pd.DataFrame(rows)
    sug.to_csv(DATA / "suggestions.csv", index=False)
    pd.DataFrame(mention_rows).to_csv(DATA / "mentions.csv", index=False)
    per = sug[sug["rank"] > 0].groupby("incident_number").size()
    print(f"{len(sample)} incidents; with options: {len(per)}; mean options {per.mean():.1f}; "
          f"model picked none: {(sug['rank'] == 0).sum()}")
    print(sug[sug["rank"] > 0]["source"].value_counts().to_string())
    log("finished")


if __name__ == "__main__":
    main()
