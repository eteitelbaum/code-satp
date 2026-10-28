"""Precompute location suggestions for the labeling app (GitHub #9).

For each incident in a split:
1. Extraction: GPT-4o-mini reads the coded summary and the full SATP entry and lists the
   places named, with each place's role (where it happened, near it, a jurisdiction, ...).
2. Candidates: each place is matched to SHRUG villages, towns, and subdistricts in the
   coded 2011 district (or, failing a close match there, elsewhere in the state, which
   catches events the coders placed in a neighboring district). Places SHRUG does not list
   closely are matched the same way against OpenStreetMap's named places (build_osm.py),
   each tied to the village or subdistrict it falls in. Options are ranked by spelling and context
   (the place's role, whether it lies in a named subdistrict, distance to a named police
   station); the weights are a first guess, to be tuned on the validation labels.
3. Selection: GPT-4o-mini sees the text and the top five options and picks one (or none),
   with a one-sentence reason.

Model calls are cached in location-labeling/data/suggest/, so a rerun only does new work,
and several incidents run at once. The app reads the outputs and calls nothing. Each run
appends its tokens, estimated cost, and any failed incidents to data/suggest/runs.jsonl
(shown in the app's sidebar). Temporary API errors are retried; if the account is out of
credit the run stops, keeping what it has done, and a rerun picks up from there.

Every run has an ID (its start time). The outputs carry it, a copy of them is kept in
data/suggest/runs/<ID>/, and each label records the run it saw, so labels made with
different runs can be told apart.

Run from the repository root after link_satp.py and build_osm.py, with OPENAI_API_KEY in .env:
    python location-labeling/suggest.py validation [--limit N]

Output (location-labeling/data/):
    suggestions.csv  one row per incident and option: rank, source, place, evidence, model pick and reason, run
    mentions.csv     one row per extracted place, with what it matched
"""

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
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
SURE = 0.92  # a SHRUG match this close needs no OpenStreetMap match
OSM_KINDS = {"village", "town", "hamlet", "locality", "police station", "camp", "other", "forest", "hill"}  # mention kinds matched to OpenStreetMap
WORKERS = 8  # incidents run at once
LOCK = threading.Lock()  # guards the cache files and token counts across workers
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
    """Lower-case letters only, with accents dropped ("Khedāntoli" -> "khedantoli")."""
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", x.lower())


def spelling(a, b, **kwargs):
    """The app's spelling similarity: average of Jaro-Winkler and an edit-distance ratio."""
    return (JaroWinkler.normalized_similarity(a, b) + fuzz.ratio(a, b) / 100) / 2


def query_names(name):
    """The name without generic words, and its parts when it joins several ("Barigada-Sashank")."""
    clean = re.sub(r"\s+", " ", re.sub(GENERIC, " ", name, flags=re.I)).strip(" .,-")
    parts = [x.strip() for x in re.split(r"[-/&,]| and ", clean) if len(norm(x)) >= 3]
    return list(dict.fromkeys([clean] + (parts if len(parts) > 1 else [])))


def fuzzy(names, keys, limit):
    """For the `limit` keys closest to each name, the best spelling similarity to any of the names."""
    return {i: max(spelling(norm(q), keys[i]) for q in names)
            for q in names for _, _, i in process.extract(norm(q), keys, scorer=spelling, limit=limit)}


def km(lat1, lon1, lat2, lon2):
    return 111.32 * math.hypot(lat1 - lat2, (lon1 - lon2) * math.cos(math.radians(lat1)))


def call_model(client, **kwargs):
    """One model call, retried on temporary errors; stops the run if the account is out of credit."""
    for wait in (5, 20, 60, None):
        try:
            r = client.chat.completions.parse(**kwargs)
            with LOCK:
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
    with LOCK, path.open("a") as f:
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

def candidates(inc, mentions, places, osm):
    """Options for one incident, best first, and the mentions with what each matched."""
    district = places[(places.state_id == inc["state_id"]) & (places.district_id == inc["district_id_2011"])]
    if district.empty:
        return [], mentions
    keys = district["name"].map(norm)
    subs = district.drop_duplicates("subdistrict_id")[["subdistrict_id", "subdistrict_name"]]
    district_names = {norm(inc["district"]), norm(inc["district_name_2011"])}
    state = places[places.state_id == inc["state_id"]]
    state_keys = state["name"].map(norm)
    district_name = state.drop_duplicates("district_id").set_index("district_id")["district_name"]
    osm_state = osm[osm.state_id == inc["state_id"]]
    osm_state_keys = osm_state["name"].map(norm)
    osm_district = osm_state[osm_state.district_id == inc["district_id_2011"]]
    osm_district_keys = osm_state_keys[osm_district.index]
    village_names = state.set_index("shrid2")["name"]

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

    def context(district_id, subdistrict_id, lat, lon):
        """Score adjustment and notes for where a matched place lies."""
        score, notes = 0.0, []
        if district_id != inc["district_id_2011"]:
            score += OTHER_DISTRICT
            name = district_name.get(district_id)  # names are listed only for coded districts
            notes.append(f"in {name.title()} district, not the coded one" if name else "in another district, not the coded one")
        if subdistrict_id in named_subs:
            score += IN_NAMED_SUBDISTRICT
            notes.append("in the named subdistrict")
        if stations and min(km(lat, lon, la, lo) for la, lo in stations) <= 15:
            score += NEAR_POLICE_STATION
            notes.append("within 15 km of the named police station")
        return score, notes

    def nearby(names, district_keys, state_keys):
        """Spelling matches in the district; if none is close (SURE), close ones elsewhere in the state."""
        hits = {i: sim for i, sim in fuzzy(names, district_keys, 8).items() if sim >= MATCH}
        if max(hits.values(), default=0) < SURE:
            hits.update({i: sim for i, sim in fuzzy(names, state_keys, 5).items()
                         if sim >= SURE and i not in district_keys.index})
        return hits

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

        # SHRUG villages and towns
        hits = nearby(names, keys, state_keys)
        best = max(hits.values(), default=0.0)
        for i, sim in sorted(hits.items(), key=lambda x: -x[1]):
            if sim < best - NEAR_SPELLING:
                continue
            p = places.loc[i]
            adjust, notes = context(p["district_id"], p["subdistrict_id"], p["lat"], p["lon"])
            options.append({"relation": relation, "source": "SHRUG", "name": p["name"], "shrid2": p["shrid2"],
                            "subdistrict_id": p["subdistrict_id"], "lat": p["lat"], "lon": p["lon"], "score": sim + weight + adjust,
                            "query": m["name"],
                            "evidence": " · ".join([f"SHRUG {p['place_type']} in {str(p['subdistrict_name']).title()} subdistrict, "
                                                    f"spelling {sim:.2f} to “{m['name']}”",
                                                    f"{m['role']} (“{m['quote']}”)"] + notes)})
        if best:
            m["matched"] = f"SHRUG {best:.2f}"

        # OpenStreetMap's named places, when SHRUG has nothing close
        if best >= SURE or m["kind"] not in OSM_KINDS or osm_state.empty:
            continue
        ohits = nearby(names, osm_district_keys, osm_state_keys)
        obest = max(ohits.values(), default=0.0)
        seen = set()
        for i, sim in sorted(ohits.items(), key=lambda x: -x[1]):
            h = osm.loc[i]
            if sim < obest - NEAR_SPELLING or h["osm_id"] in seen:
                continue
            seen.add(h["osm_id"])
            adjust, notes = context(h["district_id"], h["subdistrict_id"], h["lat"], h["lon"])
            where = f"falls in {village_names.get(h['shrid2'], h['shrid2'])}" if h["shrid2"] else "outside every village"
            alias = f" (listed as “{h['name']}”)" if h["name"] != h["primary"] else ""
            options.append({"relation": relation, "source": "OpenStreetMap", "name": h["primary"], "shrid2": h["shrid2"],
                            "subdistrict_id": h["subdistrict_id"], "lat": h["lat"], "lon": h["lon"],
                            "score": sim + weight - 0.05 + adjust,  # a little below an equal SHRUG match
                            "query": m["name"],
                            "evidence": " · ".join([f"OpenStreetMap {h['kind']}{alias}, {where}, spelling {sim:.2f} to “{m['name']}”",
                                                    f"{m['role']} (“{m['quote']}”)"] + notes)})
        if obest:
            m["matched"] += (" · " if m["matched"] else "") + f"OpenStreetMap {obest:.2f}"

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
    osm = pd.read_parquet(DATA / "osm_places.parquet")
    if args.no_osm:
        osm = osm.head(0)
    ext_cache, sel_cache = (read_jsonl(CACHE / f) for f in ("extractions.jsonl", "selections.jsonl"))

    results, failed = {}, []
    run = {"key": datetime.now().isoformat(timespec="seconds"), "split": args.split, "incidents": len(sample)}

    def log(status):
        cost = USAGE["prompt_tokens"] * PRICE["prompt"] + USAGE["completion_tokens"] * PRICE["completion"]
        append_jsonl(CACHE / "runs.jsonl", {**run, **USAGE, "estimated_cost_usd": round(cost, 4),
                                            "failed": failed, "status": status})
        print(f"{status}: {USAGE['calls']} model calls, {USAGE['prompt_tokens'] + USAGE['completion_tokens']:,} tokens, "
              f"about ${cost:.2f}; failed incidents: {len(failed)}")

    def work(inc):
        texts = satp.get(inc["incident_number"], [])
        mentions = extract(client, ext_cache, inc, texts)
        options, mentions = candidates(inc, mentions, places, osm)
        pick, reason = select(client, sel_cache, inc, texts, options) if options else (0, "no options")
        return mentions, options, pick, reason

    incidents = sample.to_dict("records")
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(work, inc): inc["incident_number"] for inc in incidents}
        for n, f in enumerate(as_completed(futures), 1):
            iid = futures[f]
            try:
                results[iid] = f.result()
            except SystemExit as e:
                pool.shutdown(cancel_futures=True)
                log(f"stopped after {n} of {len(sample)} incidents: {e}")
                sys.exit(str(e))
            except Exception as e:  # one incident's failure should not end the run
                failed.append(iid)
                print(f"{iid} failed: {e}", flush=True)
            if n % 50 == 0:
                print(f"{n} of {len(sample)}", flush=True)

    rows, mention_rows = [], []
    for inc in incidents:  # in labeling order
        iid = inc["incident_number"]
        if iid not in results:
            continue
        mentions, options, pick, reason = results[iid]
        for m in mentions:
            mention_rows.append({"incident_number": iid, **m})
        for rank, o in enumerate(options, 1):
            rows.append({"incident_number": iid, "rank": rank, **o,
                         "model_pick": rank == pick, "model_reason": reason if rank == pick else ""})
        if not options or pick == 0:
            rows.append({"incident_number": iid, "rank": 0, "relation": "", "source": "",
                         "name": "", "model_pick": True, "model_reason": reason})

    sug = pd.DataFrame(rows).assign(run=run["key"])
    sug.to_csv(DATA / "suggestions.csv", index=False)
    pd.DataFrame(mention_rows).assign(run=run["key"]).to_csv(DATA / "mentions.csv", index=False)
    keep = CACHE / "runs" / run["key"].replace(":", "-")
    keep.mkdir(parents=True, exist_ok=True)
    for f in ("suggestions.csv", "mentions.csv"):
        shutil.copy(DATA / f, keep / f)
    per = sug[sug["rank"] > 0].groupby("incident_number").size()
    print(f"{len(sample)} incidents; with options: {len(per)}; mean options {per.mean():.1f}; "
          f"model picked none: {(sug['rank'] == 0).sum()}")
    print(sug[sug["rank"] > 0]["source"].value_counts().to_string())
    log("finished")


if __name__ == "__main__":
    main()
