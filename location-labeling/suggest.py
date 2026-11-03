"""Precompute location suggestions for the labeling app (GitHub #9).

For each incident in a split:
1. Extraction: GPT-4o-mini reads the coded summary and the full SATP entry and lists the
   places named, with each place's role (where it happened, near it, a jurisdiction, ...).
   A jurisdiction is offered as the place it is named after: "under X Police Station limits" within
   the station's limits (protocol rule 2), "under X gram panchayat" near X; its subdistrict follows.
2. Candidates: each place is matched to SHRUG villages, towns, and subdistricts in the
   coded 2011 district (or, failing a close match there, elsewhere in the state, which
   catches events the coders placed in a neighboring district). Places SHRUG does not list
   closely are matched the same way against OpenStreetMap's named places (build_osm.py),
   each tied to the village or subdistrict it falls in. Options are ranked by spelling and context
   (the place's role, whether it lies in a named subdistrict, distance to a named police
   station); the weights are a first guess, to be tuned on the validation labels.
3. Where it happened: a second GPT-4o-mini call reads the same text and names the one place
   where the event happened, as written, with its relation (in, near, within police station
   limits, in a forest or area, or only a subdistrict or district) and the quote that shows it.
   It sees no options, so it decides from the text alone (the split Halterman's NGEC-2025 uses).
4. Resolution: code finds that place among the candidates, with the other places kept for
   context, and puts it first. "Between X and Y" takes the better supported end, or the
   subdistrict when the two are far apart (rule 7); a village found only under a distant
   spelling, when the text names a forest, gives way to the forest's points (rule 4).

Model calls are cached in location-labeling/data/suggest/ (a cached answer counts only for the
model that gave it), so a rerun only does new work,
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
    suggestions.csv  one row per incident and option: rank, source, place, evidence, the pick and its reason, run
    mentions.csv     one row per extracted place, with what it matched
    where.csv        one row per incident: the where-it-happened answer (place, kind, relation, other end, quote, police station)
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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
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
TOP = 8
MATCH = 0.80  # spelling similarity for a SHRUG candidate
SURE = 0.92  # a SHRUG match this close needs no OpenStreetMap match
LOOSE = 3  # a name with no match at MATCH or above gets its closest spellings instead, for the selection model to judge
FLOOR = 0.65  # ... down to this similarity
LOOSE_PENALTY = 0.3  # ranks them after the matches that pass MATCH
WORD_PENALTY = 0.15  # "Dumri" for "Dumri Vihar": ranked below a close match to the whole name
PENALTY = {"name": 0.0, "word": WORD_PENALTY, "loose": LOOSE_PENALTY}  # by how a place matched the name in the text
DESCRIPTOR_MIN, DESCRIPTOR_ALONE = 50, 0.3  # a word in this many multi-word names, found alone less than this share as often
OSM_KINDS = {"village", "town", "hamlet", "locality", "police station", "gram panchayat", "camp", "other", "forest", "hill"}  # mention kinds matched to OpenStreetMap
FOREST_KINDS = {"forest", "hill", "other"}  # mention kinds matched to mapped forests and hills (protocol rule 4); "other" catches parks
NATURAL = {"forest", "protected area", "hill", "peak", "ridge", "valley"}  # OpenStreetMap kinds they match
WORKERS = 8  # incidents run at once
LOCK = threading.Lock()  # guards the cache files and token counts across workers
PRICE = {"prompt": 0.15 / 1e6, "completion": 0.60 / 1e6}  # GPT-4o-mini, US$ per token (estimate)
BILLING = "https://platform.openai.com/settings/organization/billing/overview"
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

# ranking weights, added to spelling similarity
ROLE_WEIGHT = {"where it happened": 0.3, "near where it happened": 0.2, "jurisdiction": -0.1}
IN_NAMED_SUBDISTRICT = 0.2
NEAR_POLICE_STATION = 0.1  # within 15 km of a named police station
SETTLEMENT_KINDS = {"village", "town", "hamlet", "locality", "police station", "gram panchayat", "camp", "other"}
OTHER_DISTRICT = -0.1  # a match outside the coded district, which needs a closer spelling (SURE)
FAR_DISTRICT_KM = 150  # a village this far outside the coded district is probably another place of that name (a town is kept)
FAR_PENALTY = -1.0  # which ranks it after every other option
NEAR_SPELLING = 0.05  # keep a name's SHRUG matches within this of its best
STATION_SURE = 0.94  # a police station's namesake outside the coded district: closer than a village's (SURE), as it anchors the pick
# station names that say which of a district's stations, not where ("Sadar Police Station", "Town Police Station")
DISTRICT_POLICE = re.compile(r"\b(district|sub-?division|range|zone)\b", re.IGNORECASE)  # "Kanker District Police"
GENERIC_STATION = re.compile(r"^(sadar|town|kotwali|mufassil|mofussil|city|model|women|mahila)\b", re.IGNORECASE)

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


def is_panchayat(m):
    """A gram panchayat named X ("under X gram panchayat", "in X panchayat"), whatever kind the model gave it; not
    "a panchayat office at X", and not a police station the model called a panchayat."""
    return bool(re.search(rf"{re.escape(m['name'])}\W+(gram\W+)?panchayat", m["quote"], flags=re.I))


def is_station(m):
    """A police station named X ("under X Police Station limits"), not a village the quote ties to another station."""
    return m["kind"] == "police station" or bool(
        re.search(rf"{re.escape(m['name'])}\W+(sadar\W+|town\W+|rural\W+)?(police|p\.?s\b|thana)", m["quote"], flags=re.I))


# words that qualify a village name rather than name it ("Bade Bacheli", "Kolhua Khurd")
QUALIFIERS = {"bade", "bada", "badi", "bara", "bari", "chhota", "chota", "chhoti", "choti", "kalan", "khurd", "khas", "pina",
              "pinna", "peda", "pedda", "chinna", "buzurg", "bujurg", "uparla", "nichla", "upper", "lower", "purana", "naya",
              "alias", "tola", "toli", "para", "pada", "basti", "colony"}
PART = 0.96  # similarity given to a name found as a whole word inside a longer one: close, but not the same name
WORDS = re.compile(r"[\s\-()/,]+")
DESCRIPTORS = set(QUALIFIERS)  # with the words descriptors() finds, once candidates() has seen the names


def descriptors(places, osm):
    """Words that describe a place rather than name it, read from the names: in at least DESCRIPTOR_MIN multi-word
    SHRUG and OpenStreetMap names, and found as a whole name less than DESCRIPTOR_ALONE times as often ("Nagar",
    "Vihar", "Khera"; not "Dumri" or "Rampur"); plus QUALIFIERS, some too rare to count."""
    names = pd.concat([places["name"], osm["primary"]]).dropna().astype(str)
    alone, multi = Counter(names.map(norm)), Counter()
    for n in names:
        words = {norm(w) for w in WORDS.split(n) if len(norm(w)) >= 3}
        if len(words) > 1:
            multi.update(words)
    return {w for w, k in multi.items() if k >= DESCRIPTOR_MIN and alone[w] < DESCRIPTOR_ALONE * k} | QUALIFIERS


def load_descriptors(places, osm):
    """Fill DESCRIPTORS from the names, once."""
    with LOCK:
        if DESCRIPTORS == QUALIFIERS:
            DESCRIPTORS.update(descriptors(places, osm))


def name_parts(names):
    """The words of multi-word names as extra keys, indexed like the names ("Bade Bacheli" -> "bacheli")."""
    parts = [(i, norm(w)) for i, n in names.items() for w in WORDS.split(str(n))
             if len(re.split(r"\s+", str(n).strip())) > 1 and len(norm(w)) >= 4 and norm(w) not in DESCRIPTORS]
    return pd.Series([w for _, w in parts], index=[i for i, _ in parts], dtype=str)


def query_words(name):
    """The words of a multi-word name in the text that could name a place by themselves ("Dumri Vihar" -> ["Dumri"])."""
    words = [w for w in WORDS.split(name) if w]
    return [w for w in words if len(norm(w)) >= 4 and norm(w) not in DESCRIPTORS] if len(words) > 1 else []


def fuzzy(names, keys, limit, parts=None):
    """For the `limit` keys closest to each name, the best spelling similarity to any of the names. With `parts`,
    a name that is a whole word of a longer name also counts, at PART."""
    out = {}
    for q in names:
        for _, sim, i in process.extract(norm(q), keys, scorer=spelling, limit=limit):
            out[i] = max(out.get(i, 0.0), sim)
        if parts is not None and len(parts):
            for _, sim, i in process.extract(norm(q), parts, scorer=spelling, limit=limit):
                if sim >= 0.95:
                    out[i] = max(out.get(i, 0.0), min(sim, PART))
    return out


def station_places(names, keys, parts, district):
    """SHRUG places a police station's name points to: exact or whole-word matches if any, else near spellings
    ("Bacheli" -> Bade Bacheli, not Acheli; "Bandharia" -> Bhandaria). When some of them lie in a subdistrict
    that shares the station's name, keep only those (the Bhandaria in Bhandaria subdistrict, not the one in Kandi)."""
    hits = fuzzy(names, keys, 3, parts)
    hits = {i: sim for i, sim in hits.items() if sim >= 0.95} or {i: sim for i, sim in hits.items() if sim >= 0.88}
    named = {i: sim for i, sim in hits.items()
             if max(spelling(norm(q), norm(district.loc[i, "subdistrict_name"])) for q in names) >= 0.88}
    return named or hits


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
    kind: Literal["village", "town", "hamlet", "locality", "police station", "gram panchayat", "camp", "forest", "hill",
                  "river", "road", "mandal or block", "district", "state", "other"]
    role: Literal["where it happened", "near where it happened", "jurisdiction", "other place in this incident",
                  "another incident"]
    quote: str


class Extraction(BaseModel):
    mentions: list[Mention]


EXTRACT_PROMPT = """You help geocode incidents in India's Maoist insurgency.

The CODED SUMMARY describes one incident. The FULL SATP ENTRY, when given, is the news report it came from: it may add detail about the same incident, and it may also describe other incidents or follow-ups.

List every named place below the state level that the texts mention, once each, spelled as written. Give the proper name only, without words such as village, police station, forest, hill, camp, school, or district ("Kurdeg", not "Kurdeg Police Station"). For each place give:
- kind: what the place is (village, town, hamlet, locality, police station, gram panchayat, camp, forest, hill, river, road, mandal or block, district, state, other). A mandal, block, tehsil, taluka, or sub-division is "mandal or block". A gram panchayat or panchayat is "gram panchayat", even when it shares a village's name.
- role, relative to the incident in the CODED SUMMARY:
  - "where it happened": the event took place in or at this place
  - "near where it happened": the event took place near it, or in a forest, hill, or area named after or tied to it
  - "jurisdiction": an administrative or police area containing the spot, e.g. "under X Police Station limits", "under X gram panchayat", "X mandal", "X district"
  - "other place in this incident": named in the incident but not where it happened (where victims lived, where forces came from, where bodies were taken)
  - "another incident": belongs only to a different incident in the full entry
- quote: the shortest phrase from the text that shows the role."""


def extract(client, cache, inc, texts):
    user = f"CODED SUMMARY:\n{inc['incident_summary']}"
    if texts:
        user += "\n\nFULL SATP ENTRY:\n" + "\n\n".join(texts)
    key = inc["incident_number"] + "|" + hashlib.sha1((EXTRACT_PROMPT + user).encode()).hexdigest()[:12]  # new prompt, new call
    if key in cache and cache[key].get("model") == MODEL:
        return cache[key]["mentions"]
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

# ---- plain wording for the app (the model sees `evidence`, which stays as it is)

ABBREVIATIONS = [(r"\bW\.?L\.?S\b\.?", "Wildlife Sanctuary"), (r"\bR\.?F\b\.?", "Reserve Forest"),
                 (r"\bP\.?F\b\.?", "Protected Forest"), (r"\bN\.?P\b\.?", "National Park")]


def pretty(name):
    """Spell out abbreviations in forest and sanctuary names ("Karlapat WLS" -> "Karlapat Wildlife Sanctuary")."""
    for pattern, full in ABBREVIATIONS:
        name = re.sub(pattern, full, name)
    return name


def said(m):
    return f"The text says “{m['quote']}”."


def spelled(sim, query):
    """Label suffix when the matched name is spelled differently from the text."""
    if sim >= 0.995:
        return ""
    if sim >= SURE:
        return f" (spelled “{query}” in the text)"
    return f": a different spelling, may not be {query}"


def how_matched(sim, how, word, query):
    """For an option, by how it matched the name in the text (see ranked()): the spelling clause of its evidence,
    its label's suffix, and a sentence for why."""
    if how == "word":
        return (f"spelling {sim:.2f} to “{word}”, a word of “{query}”", f" (from “{query}” in the text)",
                f" “{query}” in the text contains the name {word}.")
    return f"spelling {sim:.2f} to “{query}”" + (", one of its closest spellings" if how == "loose" else ""), spelled(sim, query), ""


def closest(name, sim, how, source):
    """The sentence for why, when the spelling is not close."""
    if how == "loose":
        return f" {name} is one of the closest {source} names."
    return f" {name} is the closest {source} name." if sim < SURE else ""


def similarity(sim):
    return "same spelling" if sim >= 0.995 else f"spelling similarity {sim:.2f}"


def facts(notes):
    """The context notes as sentences."""
    plain = {"in the named subdistrict": "It lies in the subdistrict the text names.",
             "within 15 km of the named police station": "It is within 15 km of the police station the text names."}
    return "".join(" " + plain.get(n, f"It is {n}.") for n in notes)


class Areas:
    """Outlines of OpenStreetMap forests and protected areas, and SHRUG subdistricts, for protocol rule 4."""

    def __init__(self):
        a = pd.read_parquet(DATA / "osm_areas.parquet")
        self.outline = dict(zip(a["osm_id"], zip(shapely.from_wkb(a["wkb"].values), a["area_km2"])))
        self.subs = pd.read_parquet(DATA / "subdistrict_shapes.parquet")
        self.subs["geom"] = shapely.from_wkb(self.subs["wkb"].values)
        self._trees = {}

    def tree(self, state_id):
        if state_id not in self._trees:
            df = self.subs[self.subs.state_id == state_id].reset_index(drop=True)
            self._trees[state_id] = (df, shapely.STRtree(df["geom"].values))
        return self._trees[state_id]

    def split(self, state_id, geom):
        """Share of an outline in each subdistrict, largest first: [(subdistrict_id, share, part)]."""
        df, tree = self.tree(state_id)
        parts = [(df.loc[i, "subdistrict_id"], shapely.intersection(geom, df.loc[i, "geom"])) for i in tree.query(geom, predicate="intersects")]
        return sorted(((sid, part.area / geom.area, part) for sid, part in parts), key=lambda x: -x[1])


def candidates(inc, mentions, places, osm, areas):
    """Options for one incident, best first, and the mentions with what each matched."""
    load_descriptors(places, osm)
    district = places[(places.state_id == inc["state_id"]) & (places.district_id == inc["district_id_2011"])]
    if district.empty:
        return [], mentions
    keys = district["name"].map(norm)
    district_parts = name_parts(district["name"])
    subs = district.drop_duplicates("subdistrict_id")[["subdistrict_id", "subdistrict_name"]]
    district_names = {norm(inc["district"]), norm(inc["district_name_2011"])}
    state = places[places.state_id == inc["state_id"]]
    state_keys = state["name"].map(norm)
    state_parts = name_parts(state["name"])
    district_name = state.drop_duplicates("district_id").set_index("district_id")["district_name"]
    osm_state = osm[osm.state_id == inc["state_id"]]
    osm_state_keys = osm_state["name"].map(norm)
    settlements = osm_state[~osm_state.kind.isin(NATURAL)]  # forests and hills are matched by forest_options
    settlement_keys = osm_state_keys[settlements.index]
    settlement_district_keys = settlement_keys[settlements.district_id == inc["district_id_2011"]]
    village_names = state.set_index("shrid2")["name"]
    village_types = state.set_index("shrid2")["place_type"]
    sub_names = state.drop_duplicates("subdistrict_id").set_index("subdistrict_id")["subdistrict_name"].str.title()
    natural = osm_state[osm_state.kind.isin(NATURAL)]
    natural_keys = osm_state_keys[natural.index]
    natural_district_keys = natural_keys[natural.district_id == inc["district_id_2011"]]
    police = settlements[(settlements.kind == "police station") & (settlements.district_id == inc["district_id_2011"])]
    police_keys = settlement_keys[police.index]

    police_state = settlements[settlements.kind == "police station"]
    police_state_keys = settlement_keys[police_state.index]
    state_subs = state.drop_duplicates("subdistrict_id").dropna(subset=["subdistrict_name"])

    # police stations and blocks are looked for as villages are: in the coded district, and elsewhere in the state when
    # nothing in the district is close (SURE), since coders' districts can be wrong or have changed since 2011
    def osm_police(name):
        """OpenStreetMap police stations with the station's name ("Gomia Police Station" for "Gomia")."""
        hits = {i: sim for i, sim in fuzzy(query_names(name), police_keys, 3).items() if sim >= SURE}
        return hits or {i: sim for i, sim in fuzzy(query_names(name), police_state_keys, 3).items() if sim >= SURE}

    def station_namesakes(name):
        """SHRUG places a police station is named after (station_places)."""
        hits = station_places(query_names(name), keys, district_parts, district)
        if max(hits.values(), default=0) < SURE and not GENERIC_STATION.match(name):
            hits.update({i: sim for i, sim in station_places(query_names(name), state_keys, state_parts, state).items()
                         if sim >= STATION_SURE and i not in district.index})
        return hits

    def named_blocks(name):
        """Subdistricts with a block or mandal's name, as (subdistrict row, similarity)."""
        hits = [(s, sim) for _, s in subs.iterrows() if (sim := spelling(norm(name), norm(s["subdistrict_name"]))) >= 0.85]
        if max((sim for _, sim in hits), default=0) < SURE:
            hits += [(s, sim) for _, s in state_subs[state_subs.district_id != inc["district_id_2011"]].iterrows()
                     if (sim := spelling(norm(name), norm(s["subdistrict_name"]))) >= SURE]
        return hits

    # context: subdistricts named in the text, named police stations' locations, and the subdistricts
    # of the jurisdictions the text names ("under X Police Station", "X mandal")
    named_subs, stations, jurisdictions, jurisdiction_why = set(), [], {}, {}
    for m in mentions:
        if m["role"] == "another incident":
            continue
        if m["kind"] == "mandal or block":
            for s, _ in named_blocks(m["name"]):
                named_subs.add(s["subdistrict_id"])
                jurisdictions.setdefault(s["subdistrict_id"], f"named in the text (“{m['quote']}”)")
                jurisdiction_why.setdefault(s["subdistrict_id"], f"The text names the block or mandal (“{m['quote']}”).")
        elif m["role"] == "jurisdiction" and m["kind"] not in ("district", "state"):
            for i in station_namesakes(m["name"]):
                sid = places.loc[i, "subdistrict_id"]
                jurisdictions.setdefault(sid, f"where {m['name']} lies, a jurisdiction in the text (“{m['quote']}”)")
                jurisdiction_why.setdefault(sid, f"The text names {m['name']} as the jurisdiction (“{m['quote']}”), "
                                                 f"and {m['name']} is in {sub_names.get(sid, 'that')} subdistrict.")
        if m["kind"] == "police station":
            exact = district[keys == norm(m["name"])]
            if exact.empty:  # elsewhere in the state, only if one place has the name
                exact = state[state_keys == norm(m["name"])].pipe(lambda d: d if len(d) == 1 else d.iloc[:0])
            if len(exact):
                stations.append((exact.iloc[0]["lat"], exact.iloc[0]["lon"]))
            stations += [(osm.loc[i, "lat"], osm.loc[i, "lon"]) for i in osm_police(m["name"])]

    district_points = district[["lat", "lon"]].dropna().to_numpy()

    def from_district(lat, lon):
        """km from the coded district's nearest SHRUG place."""
        return float(111.32 * np.hypot(district_points[:, 0] - lat, (district_points[:, 1] - lon) * math.cos(math.radians(lat))).min())

    def context(district_id, subdistrict_id, lat, lon, station=False, town=False):
        """Score adjustment and notes for where a matched place lies. A match outside the coded district says how far
        outside it lies, and a village more than FAR_DISTRICT_KM away ranks last (a town or a police station's place
        keeps its rank, as the coders' district can be wrong)."""
        score, notes = 0.0, []
        if district_id != inc["district_id_2011"]:
            score += OTHER_DISTRICT
            name = district_name.get(district_id)  # names are listed only for coded districts
            notes.append(f"in {name.title()} district, not the coded one" if name else "in another district, not the coded one")
            if len(district_points) and lat is not None:
                d = from_district(lat, lon)
                notes[-1] = notes[-1].replace("not the coded one", f"{d:.0f} km from the coded one")
                if d > FAR_DISTRICT_KM and not (station or town):
                    score += FAR_PENALTY
        if subdistrict_id in named_subs:
            score += IN_NAMED_SUBDISTRICT
            notes.append("in the named subdistrict")
        if stations and min(km(lat, lon, la, lo) for la, lo in stations) <= 15:
            score += NEAR_POLICE_STATION
            notes.append("within 15 km of the named police station")
        return score, notes

    def nearby(names, district_keys, state_keys, district_parts=None, state_parts=None):
        """Spelling matches in the district; if none is close (SURE), close ones elsewhere in the state."""
        hits = {i: sim for i, sim in fuzzy(names, district_keys, 8, district_parts).items() if sim >= MATCH}
        if max(hits.values(), default=0) < SURE:
            hits.update({i: sim for i, sim in fuzzy(names, state_keys, 5, state_parts).items()
                         if sim >= SURE and i not in district_keys.index})
        return hits

    def ranked(names, words, district_keys, state_keys, district_parts=None, state_parts=None):
        """A name's matches, {index: (similarity, how, word)}: the whole name's within NEAR_SPELLING of its best ("name");
        places named by one word of it, if the word matches closely ("word": Dumri for "Dumri Vihar"); and, when nothing
        matches the whole name at all (MATCH), its LOOSE closest spellings in the district down to FLOOR ("loose"). Also
        the whole name's best similarity."""
        whole = nearby(names, district_keys, state_keys, district_parts, state_parts)
        best = max(whole.values(), default=0.0)
        hits = {i: (sim, "name", "") for i, sim in whole.items() if sim >= best - NEAR_SPELLING}

        def add(i, sim, how, word=""):
            if i not in hits or sim - PENALTY[how] > hits[i][0] - PENALTY[hits[i][1]]:
                hits[i] = (sim, how, word)
        for w in words:
            for _, sim, i in process.extract(norm(w), district_keys, scorer=spelling, limit=3):
                if sim >= SURE:
                    add(i, sim, "word", w)
        if not whole:
            for i, sim in fuzzy(names, district_keys, LOOSE).items():
                if sim >= FLOOR:
                    add(i, sim, "loose")
        return hits, best

    def forest_options(m, names, weight):
        """Points inside a mapped forest, sanctuary, or hill (protocol rule 4): in the part within a
        jurisdiction the text names (4.2), and in the middle of it (4.4)."""
        hits = nearby(names, natural_district_keys, natural_keys)
        best = max(hits.values(), default=0.0)
        out, seen = [], set()
        for i, sim in sorted(hits.items(), key=lambda x: -x[1]):
            h = osm.loc[i]
            if sim < best - NEAR_SPELLING or h["osm_id"] in seen:
                continue
            seen.add(h["osm_id"])
            base = {"relation": "area", "source": "OpenStreetMap", "osm_id": h["osm_id"], "shrid2": "", "query": m["name"]}
            role = f"{m['role']} (“{m['quote']}”)"
            # a point in the middle ranks below "near" the place it is named after (rules 4.3, 4.4)
            elsewhere = h["district_id"] != inc["district_id_2011"]
            middle = sim + weight - 0.15 + (OTHER_DISTRICT if elsewhere else 0)
            notes = context(h["district_id"], h["subdistrict_id"], h["lat"], h["lon"])[1][:1] if elsewhere else []
            if h["osm_id"] not in areas.outline:  # a peak or hill mapped as a point
                out.append({**base, "name": h["primary"], "subdistrict_id": h["subdistrict_id"], "lat": h["lat"], "lon": h["lon"],
                            "score": middle,
                            "evidence": " · ".join([f"OpenStreetMap {h['kind']} (a point), spelling {sim:.2f} to “{m['name']}”", role] + notes),
                            "label": f"A point at {pretty(h['primary'])}",
                            "why": f"{said(m)} OpenStreetMap maps the {h['kind']} as a single point (rule 4.4).{facts(notes)}",
                            "source_note": f"OpenStreetMap {h['kind']}, {similarity(sim)}" + ("." if sim >= 0.995 else f" to “{m['name']}”.")})
                continue
            geom, km2 = areas.outline[h["osm_id"]]
            split = areas.split(inc["state_id"], geom)
            shares = ", ".join(f"{share:.0%} {sub_names.get(sid, 'another subdistrict')}" for sid, share, _ in split[:3] if share >= 0.01)
            desc = f"OpenStreetMap {h['kind']}, {km2:.0f} km² ({shares}), spelling {sim:.2f} to “{m['name']}”"
            parts = [(sid, share) for sid, share, _ in split[:3] if share >= 0.01]
            source_note = (f"OpenStreetMap {h['kind']}, {km2:.0f} km²: "
                           + ", ".join(f"{share:.0%} in {sub_names.get(sid, 'another subdistrict')}" for sid, share in parts)
                           + (" subdistricts" if len(parts) > 1 else " subdistrict")
                           + ("" if sim >= 0.995 else f"; {similarity(sim)} to “{m['name']}”") + ".")
            made_part = False
            for sid, share, part in split:
                if sid in jurisdictions and share >= 0.01:
                    p = shapely.point_on_surface(part)
                    sub = sub_names.get(sid, "the named")
                    made_part = True
                    out.append({**base, "name": f"{h['primary']}, part in {sub} subdistrict", "subdistrict_id": sid,
                                "lat": p.y, "lon": p.x, "score": sim + weight + 0.15,  # above "near" the namesake (rule 4.3)
                                "evidence": " · ".join([desc, f"point in the part within {sub} subdistrict, {jurisdictions[sid]} (rule 4.2)", role]),
                                "label": f"A point in {pretty(h['primary'])}, in its {sub} subdistrict part",
                                "why": f"{said(m)} {jurisdiction_why[sid]} {sub} subdistrict holds {share:.0%} of the {h['kind']} (rule 4.2).",
                                "source_note": source_note})
            out.append({**base, "name": h["primary"], "subdistrict_id": h["subdistrict_id"], "lat": h["lat"], "lon": h["lon"],
                        "score": middle,
                        "evidence": " · ".join([desc, "point in the middle of it (rule 4.4)", role] + notes),
                        "label": f"A point in the middle of {pretty(h['primary'])}",
                        "why": f"{said(m)} The middle of the whole {h['kind']}" + (", ignoring the jurisdiction" if made_part else "")
                               + f" (rule 4.4).{facts(notes)}",
                        "source_note": source_note})
        return out, best

    options = []
    for m in mentions:
        m["matched"] = ""
        if m["role"] in ("another incident", "other place in this incident") or m["kind"] in ("district", "state") \
                or (norm(m["name"]) in district_names and m["kind"] not in SETTLEMENT_KINDS):  # keep "Ranchi" the town
            continue
        if m["role"] == "jurisdiction" and is_station(m) and DISTRICT_POLICE.search(m["name"]):
            continue  # a district's police ("Palamau District"), not a station named after a place
        if m["role"] == "jurisdiction" and m["kind"] != "mandal or block" and not is_panchayat(m):
            # "under X Police Station limits": within the station's limits, at the place it is named after
            # (protocol rule 2); other jurisdictions, near the place; then the subdistrict X lies in
            limits = is_station(m)
            namesakes = station_namesakes(m["name"])
            loose = not namesakes  # none close: the closest spelling, for the model to judge ("Gomia" -> Gumia)
            if loose:
                namesakes = {i: sim for i, sim in fuzzy(query_names(m["name"]), keys, 1).items() if sim >= FLOOR}
            for i, sim in namesakes.items():
                p = places.loc[i]
                sub = str(p["subdistrict_name"]).title()
                adjust, notes = context(p["district_id"], p["subdistrict_id"], p["lat"], p["lon"], station=True)
                adjust -= LOOSE_PENALTY if loose else 0
                options.append({"relation": "limits" if limits else "near", "source": "SHRUG", "name": p["name"],
                                "shrid2": p["shrid2"], "subdistrict_id": p["subdistrict_id"], "lat": p["lat"], "lon": p["lon"],
                                "score": sim + ROLE_WEIGHT["jurisdiction"] + 0.01 + adjust,  # just above its subdistrict
                                "query": m["name"], "match": "loose" if loose else "name",
                                "evidence": " · ".join([f"SHRUG {p['place_type']} in {sub} subdistrict, spelling {sim:.2f} to “{m['name']}”"
                                                        + (", the closest spelling" if loose else ""),
                                                        f"jurisdiction (“{m['quote']}”)"] + notes),
                                "label": (f"Within {m['name']} police station limits: {p['name']} {p['place_type']}, {sub} subdistrict"
                                          f"{spelled(sim, m['name'])}"
                                          if limits else f"Near {p['name']} {p['place_type']}, {sub} subdistrict{spelled(sim, m['name'])}"),
                                "why": said(m) + (" That names the police station's area, so the label is the place the station is "
                                                  "named after (rule 2)." if limits else " That names an area around this place.")
                                       + facts(notes),
                                "source_note": f"SHRUG {p['place_type']} in {sub} subdistrict, {similarity(sim)}"
                                               + ("." if sim >= 0.995 else f" to “{m['name']}”.")})
                options.append({"relation": "subdistrict", "source": "SHRUG",
                                "name": str(p["subdistrict_name"]).title(), "shrid2": "",
                                "subdistrict_id": p["subdistrict_id"], "lat": None, "lon": None,
                                "score": sim + ROLE_WEIGHT["jurisdiction"] - (LOOSE_PENALTY if loose else 0), "query": m["name"],
                                "evidence": f"subdistrict of {p['name']}, a jurisdiction in the text (“{m['quote']}”)",
                                "label": f"{str(p['subdistrict_name']).title()} subdistrict",
                                "why": f"{said(m)} That names a jurisdiction, and {p['name']} is in {str(p['subdistrict_name']).title()} subdistrict (rule 2).",
                                "source_note": "SHRUG subdistrict" + ("" if sim >= 0.995 else f"; {p['name']} has {similarity(sim)} to “{m['name']}”") + "."})
                m["matched"] = f"SHRUG {sim:.2f} (jurisdiction)"
            # the station itself, where OpenStreetMap has it: the map point (rule 2)
            for i, sim in osm_police(m["name"]).items() if limits else []:
                h = osm.loc[i]
                adjust, notes = context(h["district_id"], h["subdistrict_id"], h["lat"], h["lon"], station=True)
                where = f"falls in {village_names.get(h['shrid2'], h['shrid2'])}" if h["shrid2"] else "outside every village"
                options.append({"relation": "limits", "source": "OpenStreetMap", "name": h["primary"], "osm_id": h["osm_id"],
                                "shrid2": h["shrid2"], "subdistrict_id": h["subdistrict_id"], "lat": h["lat"], "lon": h["lon"],
                                "score": sim + ROLE_WEIGHT["jurisdiction"] - 0.05 + adjust, "query": m["name"],
                                "evidence": " · ".join([f"OpenStreetMap police station, {where}, spelling {sim:.2f} to “{m['name']}”",
                                                        f"jurisdiction (“{m['quote']}”)"] + notes),
                                "label": f"Within {m['name']} police station limits: the station's point ({h['primary']})",
                                "why": said(m) + " OpenStreetMap maps the station; its point is the map point, and the place is "
                                                 "the one the station is named after (rule 2)." + facts(notes),
                                "source_note": f"OpenStreetMap police station, {where}, {similarity(sim)}"
                                               + ("." if sim >= 0.995 else f" to “{m['name']}”.")})
                m["matched"] += (" · " if m["matched"] else "") + f"OpenStreetMap {sim:.2f} (police station)"
            # SHRUG has nothing close to the station's name: the OpenStreetMap town or village it is named after, as
            # for villages (rules 2, 5)
            seen = set()
            for i, sim in nearby(query_names(m["name"]), settlement_district_keys, settlement_keys).items() \
                    if limits and (loose or max(namesakes.values()) < SURE) else []:
                h = osm.loc[i]
                if sim < SURE or h["kind"] == "police station" or h["osm_id"] in seen:
                    continue
                seen.add(h["osm_id"])
                adjust, notes = context(h["district_id"], h["subdistrict_id"], h["lat"], h["lon"], station=True)
                where = f"falls in {village_names.get(h['shrid2'], h['shrid2'])}" if h["shrid2"] else "outside every village"
                kind = h["kind"].replace("_", " ")
                options.append({"relation": "limits", "source": "OpenStreetMap", "name": h["primary"], "osm_id": h["osm_id"],
                                "shrid2": h["shrid2"], "subdistrict_id": h["subdistrict_id"], "lat": h["lat"], "lon": h["lon"],
                                "score": sim + ROLE_WEIGHT["jurisdiction"] - 0.05 + adjust, "query": m["name"],
                                "evidence": " · ".join([f"OpenStreetMap {kind}, {where}, spelling {sim:.2f} to “{m['name']}”",
                                                        f"jurisdiction (“{m['quote']}”)"] + notes),
                                "label": f"Within {m['name']} police station limits: {h['primary']} {kind} (OpenStreetMap)",
                                "why": said(m) + f" SHRUG has no place close to the station's name, so OpenStreetMap's {kind} "
                                                 "is the place it is named after, and its point is the map point (rules 2, 5)."
                                       + facts(notes),
                                "source_note": f"OpenStreetMap {kind}, {where}, {similarity(sim)}"
                                               + ("." if sim >= 0.995 else f" to “{m['name']}”.")})
                m["matched"] += (" · " if m["matched"] else "") + f"OpenStreetMap {sim:.2f} ({kind})"
            continue
        # "under X gram panchayat": near the place it is named after (protocol rule 2)
        panchayat = is_panchayat(m)
        weight = ROLE_WEIGHT["near where it happened"] if panchayat else ROLE_WEIGHT.get(m["role"], 0.0)
        relation = "in" if m["role"] == "where it happened" and m["kind"] in SETTLEMENT_KINDS and not panchayat else "near"
        if m["kind"] == "mandal or block":
            for s, sim in named_blocks(m["name"]):
                other = s.get("district_id", inc["district_id_2011"]) != inc["district_id_2011"]
                elsewhere = f" in {s['district_name'].title()} district, not the coded one" if other else ""
                options.append({"relation": "subdistrict", "source": "SHRUG", "name": s["subdistrict_name"].title(),
                                "shrid2": "", "subdistrict_id": s["subdistrict_id"], "lat": None, "lon": None,
                                "score": sim + 0.1 * (weight + 0.1) + (OTHER_DISTRICT if other else 0), "query": m["name"],
                                "evidence": f"subdistrict named in the text (“{m['quote']}”), spelling {sim:.2f}"
                                            + (f" · {elsewhere.strip()}" if other else ""),
                                "label": f"{s['subdistrict_name'].title()} subdistrict" + (f" ({s['district_name'].title()} district)" if other else ""),
                                "why": f"{said(m)} That names the block or mandal." + (f" It is{elsewhere}." if other else ""),
                                "source_note": "SHRUG subdistrict" + ("" if sim >= 0.995 else f", {similarity(sim)} to “{m['name']}”") + "."})
                m["matched"] = "SHRUG subdistrict"
            continue
        names = query_names(m["name"])
        words = query_words(names[0])

        # SHRUG villages and towns
        hits, best = ranked(names, words, keys, state_keys, district_parts, state_parts)
        for i, (sim, how, word) in sorted(hits.items(), key=lambda x: PENALTY[x[1][1]] - x[1][0]):
            p = places.loc[i]
            adjust, notes = context(p["district_id"], p["subdistrict_id"], p["lat"], p["lon"], town=p["place_type"] == "town")
            clause, suffix, contains = how_matched(sim, how, word, m["name"])
            options.append({"relation": relation, "source": "SHRUG", "name": p["name"], "shrid2": p["shrid2"],
                            "subdistrict_id": p["subdistrict_id"], "lat": p["lat"], "lon": p["lon"],
                            "score": sim - PENALTY[how] + weight + adjust, "query": m["name"], "match": how, "spelling": sim,
                            "evidence": " · ".join([f"SHRUG {p['place_type']} in {str(p['subdistrict_name']).title()} subdistrict, {clause}",
                                                    f"{m['role']} (“{m['quote']}”)"] + notes),
                            "label": f"{relation.capitalize()} {p['name']} {p['place_type']}, "
                                     f"{str(p['subdistrict_name']).title()} subdistrict{suffix}",
                            "why": said(m) + contains
                                   + (f" The {m['kind']} is named after this place (rule 4.3)." if m["kind"] in ("forest", "hill") else "")
                                   + (" The gram panchayat is named after this place (rule 2)." if panchayat else "")
                                   + closest(p["name"], sim, how, "SHRUG") + facts(notes),
                            "source_note": f"SHRUG {p['place_type']} in {str(p['subdistrict_name']).title()} subdistrict, {similarity(sim)}"
                                           + ("." if sim >= 0.995 else f" to “{word or m['name']}”.")})
        if hits:
            sim, how, word = max(hits.values(), key=lambda h: h[0])
            m["matched"] = f"SHRUG {sim:.2f}" + (f" (the word {word})" if how == "word" else "")

        # forests, hills, sanctuaries: points inside the mapped area, whatever SHRUG found
        if m["kind"] in FOREST_KINDS and not osm_state.empty:
            found, fbest = forest_options(m, names, weight)
            options += found
            if found:
                m["matched"] += (" · " if m["matched"] else "") + f"OpenStreetMap area {fbest:.2f}"

        # OpenStreetMap's named places, when SHRUG has nothing close in the coded district (SHRUG's Dhurwa in Dumka
        # must not hide OpenStreetMap's Dhurwa police station in Ranchi)
        local = max((sim for i, (sim, how, _) in hits.items() if how == "name"
                     and places.at[i, "district_id"] == inc["district_id_2011"]), default=0.0)
        if local >= SURE or m["kind"] not in OSM_KINDS or osm_state.empty:
            continue
        ohits, _ = ranked(names, words, settlement_district_keys, settlement_keys)
        seen = set()
        for i, (sim, how, word) in sorted(ohits.items(), key=lambda x: PENALTY[x[1][1]] - x[1][0]):
            h = osm.loc[i]
            if h["osm_id"] in seen:
                continue
            seen.add(h["osm_id"])
            adjust, notes = context(h["district_id"], h["subdistrict_id"], h["lat"], h["lon"], town=h["kind"] in ("town", "city"))
            where = f"falls in {village_names.get(h['shrid2'], h['shrid2'])}" if h["shrid2"] else "outside every village"
            alias = f" (listed as “{h['name']}”)" if h["name"] != h["primary"] else ""
            clause, suffix, contains = how_matched(sim, how, word, m["name"])
            options.append({"relation": relation, "source": "OpenStreetMap", "name": h["primary"], "osm_id": h["osm_id"],
                            "shrid2": h["shrid2"],
                            "subdistrict_id": h["subdistrict_id"], "lat": h["lat"], "lon": h["lon"],
                            "score": sim - PENALTY[how] + weight - 0.05 + adjust,  # a little below an equal SHRUG match
                            "query": m["name"], "match": how, "spelling": sim,
                            "evidence": " · ".join([f"OpenStreetMap {h['kind']}{alias}, {where}, {clause}",
                                                    f"{m['role']} (“{m['quote']}”)"] + notes),
                            "label": f"{relation.capitalize()} {h['primary']}"
                                     + ("" if re.search(r"station|halt", h["primary"], flags=re.I) else f" {h['kind'].replace('_', ' ')}")
                                     + (f", {sub_names[h['subdistrict_id']]} subdistrict" if h["subdistrict_id"] in sub_names else "")
                                     + suffix,
                            "why": said(m) + contains + closest(h["primary"], sim, how, "OpenStreetMap") + facts(notes),
                            "source_note": f"OpenStreetMap {h['kind'].replace('_', ' ')}"
                                           + (f", listed as “{h['name']}”" if alias else "")
                                           + (f", inside SHRUG {village_types.get(h['shrid2'], 'village')} {village_names.get(h['shrid2'], h['shrid2'])}" if h["shrid2"]
                                              else ", outside every SHRUG village")
                                           + f", {similarity(sim)}" + ("." if sim >= 0.995 else f" to “{word or m['name']}”.")})
        obest = max((sim for sim, how, _ in ohits.values() if how != "loose"), default=0.0)  # a loose spelling is not a find
        if obest:
            m["matched"] += (" · " if m["matched"] else "") + f"OpenStreetMap {obest:.2f}"

    # an OpenStreetMap village that SHRUG already offers under the same name within 5 km is the same place
    shrug = [(norm(o["name"]), o["lat"], o["lon"]) for o in options if o["source"] == "SHRUG" and o["lat"] is not None]
    options = [o for o in options if not (o["source"] == "OpenStreetMap" and o["relation"] in ("in", "near") and any(
        n == norm(o["name"]) and km(o["lat"], o["lon"], la, lo) <= 5 for n, la, lo in shrug))]

    # one option per place: keep the best-scoring
    seen, out = set(), []
    for o in sorted(options, key=lambda o: -o["score"]):
        k = (o["source"], o["shrid2"] or o["subdistrict_id"] if o["source"] == "SHRUG" else (round(o["lat"], 4), round(o["lon"], 4)))
        if k not in seen:
            seen.add(k)
            out.append(o)
    out = out[:TOP]

    # fewer than five: add the subdistricts the suggested places lie in, for when the place itself is uncertain (rule 7)
    offered = {o["subdistrict_id"] for o in out if o["relation"] == "subdistrict"}
    for rank, o in enumerate(list(out), 1):
        sid = o["subdistrict_id"]
        if len(out) >= TOP or o["relation"] == "subdistrict" or o.get("match") == "loose" or sid in offered or sid not in sub_names:
            continue
        offered.add(sid)
        out.append({"relation": "subdistrict", "source": "SHRUG", "name": sub_names[sid], "shrid2": "",
                    "subdistrict_id": sid, "lat": None, "lon": None, "score": 0.0, "query": "",
                    "evidence": f"subdistrict containing option {rank}, if that place is uncertain",
                    "label": f"{sub_names[sid]} subdistrict",
                    "why": f"The subdistrict option {rank} lies in, if you cannot place the event that closely (rule 7).",
                    "source_note": "SHRUG subdistrict."})
    return out, mentions


# ---------------------------------------------------------------- 3. where it happened

WHERE_PROMPT = """You help geocode incidents in India's Maoist insurgency.

The CODED SUMMARY describes one incident. The FULL SATP ENTRY, when given, is the news report it came from. Use it for detail about the same incident, and ignore any other incidents it describes.

Say where the event in the CODED SUMMARY happened, as precisely as the texts allow. Give:
- place: the most specific named place where the event happened, spelled as written, without words such as village, police station, forest, or district ("Kurdeg", not "Kurdeg Police Station"). Leave it empty if the texts name nothing below the district. For "within police station limits," give the station's name as the place.
- kind: village, town, hamlet, locality, police station, camp, forest, hill, road, gram panchayat, mandal or block, other
- relation:
  - "in": the event happened in or at the place. This includes an event at a police station, police post, outpost, camp, market, or railway station named after the place, and a surrender before X Police or at X police headquarters, which is in X.
  - "near": the event happened near the place ("near X", "adjoining X", "outskirts of X", "in the forest near X village", "in the jungles of X", a road to or through X, between X and another place), in a forest or area named after or tied to X (such as one under X gram panchayat), or under X gram panchayat.
  - "within police station limits": the texts say only that it happened under X Police Station (limits or area) and name nothing more specific.
  - "in a forest or area": in a named forest, hill, or sanctuary, when the texts name no village or town in or near it.
  - "subdistrict" or "district": the texts name only a block, mandal, tehsil, or district.
- other_end: for "between X and Y" or an "X-Y road," the second place (Y); otherwise empty.
- quote: the shortest phrase from the texts that shows the place and the relation.
- police_station: the police station whose limits the texts place the event under, if any.

Rules:
- Give where the event happened, not where victims lived, where forces came from, or where bodies were taken.
- A village or town beats a police station, forest, or district named in the same phrase: "near Maharbeda village under Jhara Police Station limits" is near Maharbeda, with police station Jhara."""


class Where(BaseModel):
    place: str
    kind: Literal["village", "town", "hamlet", "locality", "police station", "camp", "forest", "hill", "road",
                  "gram panchayat", "mandal or block", "other"]
    relation: Literal["in", "near", "within police station limits", "in a forest or area", "subdistrict", "district"]
    other_end: str
    quote: str
    police_station: str


def where(client, cache, inc, texts):
    user = f"CODED SUMMARY:\n{inc['incident_summary']}"
    if texts:
        user += "\n\nFULL SATP ENTRY:\n" + "\n\n".join(texts)
    key = inc["incident_number"] + "|" + hashlib.sha1((MODEL + WHERE_PROMPT + user).encode()).hexdigest()[:12]
    if key in cache:
        return cache[key]["where"]
    r = call_model(
        client, model=MODEL, temperature=0, response_format=Where,
        messages=[{"role": "system", "content": WHERE_PROMPT}, {"role": "user", "content": user}],
    )
    w = r.choices[0].message.parsed.model_dump()
    row = {"key": key, "model": MODEL, "where": w}
    append_jsonl(CACHE / "where.jsonl", row)
    cache[key] = row
    return w


# ---------------------------------------------------------------- 4. resolution

# the role a where-it-happened relation gives the place when it is looked up
WHERE_ROLE = {"in": "where it happened", "near": "near where it happened", "within police station limits": "jurisdiction",
              "in a forest or area": "near where it happened", "subdistrict": "where it happened"}
CLOSE_KM = 5  # "between X and Y": ends farther apart than this, in one subdistrict, give the subdistrict (rule 7)
STATION_WORDS = re.compile(r"\b(police stations?|police outposts?|police posts?|police|thana|p\.?s|outposts?|posts?|camp)\b\.?",
                           re.IGNORECASE)
INSIDE_KM = 15  # this close to the named police station, a match counts as inside its limits
FAR_KM = 25  # a match farther than this from the named police station or subdistrict is taken for another place


def jurisdiction(inc, station, mentions, options, places):
    """The area the text places the event under: the subdistricts of the police station (where its namesake lies, and
    any subdistrict named like it) and of a block or mandal the text names, and the station's points. Returns the
    station's limits options found by name, whether the station was found, the area's name for notes, and a function
    giving a match's distance outside the area in km (0 inside it, None with no area)."""
    state = places[places.state_id == inc["state_id"]]
    if DISTRICT_POLICE.search(station) or GENERIC_STATION.match(station) \
            or norm(station) in {norm(inc["district"]), norm(inc["district_name_2011"])}:  # a district's police, not a station
        station = ""
    limits = [o for o in options if station and o["relation"] == "limits" and o["query"] == station
              and o.get("match", "name") == "name" and ": a different spelling" not in o["label"] and o["lat"] is not None]
    subs = {o["subdistrict_id"] for o in limits}
    if station:
        named = state.drop_duplicates("subdistrict_id").dropna(subset=["subdistrict_name"])
        subs |= set(named.loc[named.subdistrict_name.map(lambda s: spelling(norm(station), norm(s)) >= 0.95), "subdistrict_id"])
    found = bool(subs)
    blocks = {m["name"] for m in mentions if m["kind"] == "mandal or block" and m["role"] != "another incident"}
    named_blocks = [o for o in options if o["relation"] == "subdistrict" and o["query"] in blocks]
    subs |= {o["subdistrict_id"] for o in named_blocks}
    if not subs:
        return limits, False, "", lambda o: None
    stations = [(o["lat"], o["lon"]) for o in limits]
    edge = [tuple(p) for p in state.loc[state.subdistrict_id.isin(subs), ["lat", "lon"]].dropna().to_numpy()]
    name = f"the {station} police station area" if found else f"{named_blocks[0]['name']} subdistrict"

    def away(o):
        if o["relation"] not in ("in", "near") or o["lat"] is None or o["subdistrict_id"] in subs:
            return 0.0
        if stations and min(km(o["lat"], o["lon"], la, lo) for la, lo in stations) <= INSIDE_KM:
            return 0.0
        return min(km(o["lat"], o["lon"], la, lo) for la, lo in stations + edge)
    return limits, found, name, away


def resolve(inc, mentions, w, places, osm, areas):
    """The where-it-happened place among the candidates, with step 1's other places kept for context (named subdistricts,
    stations, jurisdictions). Returns the options with the pick first, the mentions with what they matched,
    the pick's rank (1, or 0 for none), and the reason."""
    w = dict(w)
    station = re.sub(r"\s+", " ", STATION_WORDS.sub(" ", w["police_station"])).strip(" .,")
    if w["relation"] == "within police station limits":
        if not w["place"]:  # the station given only in its own field
            w["place"] = station
        elif station and norm(station) != norm(w["place"]):  # a place under the station's limits is that place (rule 2)
            w["relation"] = "in" if w["kind"] in ("village", "town", "hamlet", "locality", "camp") else "near"
    if not w["place"] and w["other_end"]:  # "connecting X and Y" given as only Y: X is step 1's place first named in the quote
        def at(m):
            return re.search(rf"\b{re.escape(m['name'])}\b", w["quote"], flags=re.I)
        named = sorted((m for m in mentions if norm(m["name"]) != norm(w["other_end"]) and m["role"] != "another incident"
                        and at(m)), key=lambda m: at(m).start())
        if named:
            w["place"], w["kind"] = named[0]["name"], named[0]["kind"]
        else:
            w["place"], w["other_end"] = w["other_end"], ""
    between = re.search(r"\bbetween\b", w["quote"], flags=re.I)
    if w["place"] and w["other_end"] and between and not re.search(rf"\b{re.escape(w['place'])}\b", w["quote"][between.end():], flags=re.I):
        w["other_end"] = ""  # "the forest of Naingunda between Kotmi and Kasansoor": the place is not one of the ends
    if w["place"] and re.search(rf"\b{re.escape(w['place'])}\s+(block|mandal|tehsil|taluka?)\b", w["quote"], flags=re.I):
        w["relation"] = "subdistrict"  # "at Churchu block": the block, not a village of that name
    said_where = f"{w['relation']} {w['place']}".strip() + (f" (“{w['quote']}”)" if w["quote"] else "")
    if not w["place"] or w["relation"] == "district":
        options, mentions = candidates(inc, mentions, places, osm, areas)
        why = "The texts name nothing below the district" if w["relation"] == "district" else "The model named no place"
        return options, mentions, 0, f"{why}: {said_where}."

    names = [n for n in (w["place"], w["other_end"]) if n]
    keep = [dict(m) for m in mentions if norm(m["name"]) not in {norm(n) for n in names}]
    kind = {"within police station limits": "police station", "subdistrict": "mandal or block"}.get(w["relation"], w["kind"])
    if w["relation"] == "in a forest or area" and kind not in FOREST_KINDS:
        kind = "forest"
    event = [{"name": w["place"], "kind": kind, "role": WHERE_ROLE[w["relation"]], "quote": w["quote"]}]
    if w["other_end"]:
        event.append({"name": w["other_end"], "kind": "village", "role": "near where it happened", "quote": w["quote"]})
    if station and norm(station) not in {norm(m["name"]) for m in keep + event}:
        keep.append({"name": station, "kind": "police station", "role": "jurisdiction",
                     "quote": f"under {w['police_station']}"})
    options, mentions = candidates(inc, keep + event, places, osm, areas)

    # the police station or block the text names: a place's matches inside it come first, and one outside it says how far
    limits, by_station, area_name, away = jurisdiction(inc, station, mentions, options, places)
    for o in options:
        if o["query"] in names and (d := away(o)):
            o["evidence"] += f" · {d:.0f} km from {area_name}"
            o["why"] += f" It is {d:.0f} km from {area_name}."

    coded = places[(places.state_id == inc["state_id"]) & (places.district_id == inc["district_id_2011"])][["lat", "lon"]].dropna().to_numpy()
    far_off = {}  # a name found only more than FAR_DISTRICT_KM outside the coded district: (its best option, km)

    def from_district(o):
        """A village option's km from the coded district's nearest SHRUG place (0 for a town, as for context())."""
        if o["relation"] not in ("in", "near") or o["lat"] is None or not len(coded) \
                or re.match(r"(SHRUG|OpenStreetMap) (town|city)\b", o["evidence"]):
            return 0.0
        return float(111.32 * np.hypot(coded[:, 0] - o["lat"], (coded[:, 1] - o["lon"]) * math.cos(math.radians(o["lat"]))).min())

    def best_for(name):
        """A name's best option: a SHRUG place chosen by name, a point only for a forest (rules 4, 5), inside the
        text's police station or block area if any is, and not far outside the coded district."""
        mine = [o for o in options if o["query"] == name and (o["relation"] == "subdistrict") == (w["relation"] == "subdistrict")]
        close = [o for o in mine if from_district(o) <= FAR_DISTRICT_KM]
        if mine and not close:
            far_off[name] = (mine[0], from_district(mine[0]))
        mine = close
        # a closest spelling gives way to a close match of the whole name in either gazetteer ("Chambannur":
        # OpenStreetMap's Champannur over SHRUG's Amballur)
        if any(o.get("match") == "name" and o.get("spelling", 0) >= SURE for o in mine):
            mine = [o for o in mine if o.get("match") != "loose"]
        if w["relation"] != "in a forest or area" and any(o["source"] == "SHRUG" for o in mine):
            mine = [o for o in mine if o["source"] == "SHRUG"]
        inside = [o for o in mine if not away(o)]
        return (inside or mine)[0] if mine else None

    best = {n: o for n in names if (o := best_for(n))}
    note = ""
    if not best and w["place"] in far_off and by_station and limits:  # found only far away, under a named station
        o, d = far_off[w["place"]]
        best = {w["place"]: limits[0]}
        note = (f" {w['place']} was found only {d:.0f} km from the coded district ({o['label']}), so probably not the place;"
                f" the station's limits (rule 2).")
    if not best:  # a landmark no gazetteer lists ("Koyataad Railway Crossing of Dumri Vihar"): a place its quote names
        named = [m["name"] for m in mentions if m["name"] not in names
                 and re.search(rf"\b{re.escape(m['name'])}\b", w["quote"], flags=re.I)]
        found = [o for o in map(best_for, named) if o]
        if not found:
            if w["place"] in far_off:
                o, d = far_off[w["place"]]
                return options, mentions, 0, (f"The model says: {said_where}. {w['place']} was found only {d:.0f} km from "
                                              f"the coded district ({o['label']}), so probably not the place.")
            return options, mentions, 0, f"{w['place']} was not found: {said_where}."
        fallback = max(found, key=lambda o: o["score"])
        best = {w["place"]: fallback}
        note = f" {w['place']} was not found; the quote also names {fallback['query']}."
    a, b = best.get(w["place"]), best.get(w["other_end"])
    if a and b and a.get("lat") and b.get("lat"):
        far = km(a["lat"], a["lon"], b["lat"], b["lon"])
        sub = next((o for o in options if o["relation"] == "subdistrict" and o["subdistrict_id"] == a["subdistrict_id"]), None)
        if far > CLOSE_KM and a["subdistrict_id"] == b["subdistrict_id"] and sub:
            pick = sub
            note += f" {w['place']} and {w['other_end']} are {far:.0f} km apart, so their subdistrict (rule 7)."
        else:
            pick = max((a, b), key=lambda o: o["score"])
            other = b if pick is a else a
            note += f" The other end, {other['name']}, is {far:.0f} km away (rule 7)."
    else:
        pick = a or b
        # rule 4.1 holds only for a village that can be found: a distant spelling, when the text names a forest,
        # gives way to the forest's points (rules 4.2, 4.4)
        area = next((o for o in options if o["relation"] == "area"), None)
        if pick["relation"] in ("in", "near") and ": a different spelling" in pick["label"] and area \
                and any(m["kind"] in ("forest", "hill") for m in mentions):
            pick = area
            note += f" {w['place']} matches only a distant spelling, so the forest (rule 4)."
        elif by_station and (d := away(pick)) and d > FAR_KM:  # far outside the station's area: another place of that name
            found = f" {w['place']} was found only {d:.0f} km from {area_name} ({pick['label']}), so probably not the place"
            if limits:
                pick, note = limits[0], note + found + "; the station's limits (rule 2)."
            else:
                return options, mentions, 0, f"The model says: {said_where}.{note}{found}."

    # the text names a station after a village beside it ("Dumri Vihar": Dumri Bihar station, Dumri village): the
    # place is the village, the station is the map point (rule 2); it follows the pick in the list
    stop = next((o for o in options if o["query"] == pick.get("query") and o["source"] == "OpenStreetMap"
                 and o.get("match") == "name" and re.match(r"OpenStreetMap (railway|police) station", o["evidence"])), None)
    # the pick is a railway or police station itself ("Titilagarh Railway Station"): the place is the SHRUG village or
    # town it sits in or is named after, the station is the map point (rule 2)
    named_after = None
    if pick["source"] == "OpenStreetMap" and re.match(r"OpenStreetMap (railway|police) station", pick["evidence"]):
        words = set(norm(pick["name"]).split())
        named_after = next((o for o in options if o["source"] == "SHRUG" and o["relation"] in ("in", "near") and o.get("lat")
                            and ((pick["shrid2"] and o["shrid2"] == pick["shrid2"])
                                 or (norm(o["name"]) in words and km(o["lat"], o["lon"], pick["lat"], pick["lon"]) <= CLOSE_KM))),
                           None)
        if named_after:
            pick, stop = named_after, pick
    head = [pick]
    if stop and pick["source"] == "SHRUG" and (named_after or pick.get("match") != "name"):
        head = [{**pick, "point_lat": stop["lat"], "point_lon": stop["lon"], "point_name": stop["name"]}, stop]
        note += f" Map point: {stop['name']} (rule 2)."
    rest = [o for o in options if o is not pick and o is not stop]
    return head + rest, mentions, 1, f"The model says: {said_where}.{note}"


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
    areas = Areas()
    load_descriptors(places, osm)
    for state_id in sample["state_id"].unique():  # build the subdistrict indexes before the workers share them
        areas.tree(state_id)
    if args.no_osm:
        osm = osm.head(0)
    ext_cache, where_cache = (read_jsonl(CACHE / f) for f in ("extractions.jsonl", "where.jsonl"))

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
        w = where(client, where_cache, inc, texts)
        options, mentions, pick, reason = resolve(inc, mentions, w, places, osm, areas)
        return mentions, options, pick, reason, w

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

    rows, mention_rows, where_rows = [], [], []
    for inc in incidents:  # in labeling order
        iid = inc["incident_number"]
        if iid not in results:
            continue
        mentions, options, pick, reason, w = results[iid]
        where_rows.append({"incident_number": iid, **w})
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
    pd.DataFrame(where_rows).assign(run=run["key"]).to_csv(DATA / "where.csv", index=False)
    keep = CACHE / "runs" / run["key"].replace(":", "-")
    keep.mkdir(parents=True, exist_ok=True)
    for f in ("suggestions.csv", "mentions.csv", "where.csv"):
        shutil.copy(DATA / f, keep / f)
    per = sug[sug["rank"] > 0].groupby("incident_number").size()
    print(f"{len(sample)} incidents; with options: {len(per)}; mean options {per.mean():.1f}; "
          f"no place picked: {(sug['rank'] == 0).sum()}")
    print(sug[sug["rank"] > 0]["source"].value_counts().to_string())
    log("finished")


if __name__ == "__main__":
    main()
