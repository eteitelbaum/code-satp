"""Location labeling app: assign each sampled incident to a Census 2011 place (SHRUG).

Run from the repository root, after build_lookup.R and draw_sample.py:
    streamlit run location-labeling/app.py

For each incident, record the most precise unit you are confident of: a village or
town (SHRUG ID), otherwise the subdistrict, otherwise the district. Labels autosave
to location-labeling/labels/labels.csv, one row per incident, so you can stop and
resume at any time.
"""

import html
import json
import math
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import pydeck as pdk
import shapely
import streamlit as st
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

ROOT = Path(__file__).parent
SAMPLE_FILE = ROOT / "data" / "sample.csv"
PLACES_FILE = ROOT / "data" / "places.parquet"
SHAPES_FILE = ROOT / "data" / "place_shapes.parquet"
AREAS_FILE = ROOT / "data" / "osm_areas.parquet"
SUBDIST_FILE = ROOT / "data" / "subdistrict_shapes.parquet"
SUGGEST_FILE = ROOT / "data" / "suggestions.csv"  # from suggest.py
MENTIONS_FILE = ROOT / "data" / "mentions.csv"
RUNS_FILE = ROOT / "data" / "suggest" / "runs.jsonl"  # suggest.py runs: tokens, cost, failures
BILLING = "https://platform.openai.com/settings/organization/billing/overview"
LABELS_FILE = ROOT / "labels" / "labels.csv"
LINKS_FILE = ROOT / "data" / "satp_links.csv"  # from link_satp.py
ENTRIES_FILE = ROOT / "data" / "satp_entries.csv"  # from fetch_satp.py

PRECISIONS = ["in a village or town", "near a village or town", "in a forest or area", "subdistrict", "district", "state or unknown"]
AT_PLACE = PRECISIONS[:2]  # precisions that name a SHRUG place
AREA = PRECISIONS[2]  # a point inside a forest, sanctuary, or hill (protocol rule 4); the unit comes from the point
LEGACY = {"village or town": "in a village or town"}  # earlier label values
NAMED = ["yes", "no"]  # does the report name a place below the district for where it happened? (#17)
CERTAINTY = ["sure", "probable", "guess"]  # how sure the label is the right place, apart from its size (#17)
PLACE_TYPES = ["village", "town", "police station", "camp", "forest", "hill", "river", "road", "other"]
REASONS = ["", "hamlet", "local name", "area only", "camp", "not found", "other"]
FOUND_VIA = ["suggestion", "shortlist", "search", "Google Maps", "OpenStreetMap", "SATP page", "pin"]
SKIP_REASONS = ["", "not a single incident", "text does not describe the coded event", "other (say why in the notes)"]
LABEL_COLUMNS = [
    "incident_number", "split", "order", "named_below_district", "precision", "certainty", "shrid2", "place_name", "subdistrict_id",
    "subdistrict_name", "state_id", "district_id_2011", "place_type", "reason", "found_via",
    "preselected", "preselect_source", "confirmed", "pin_consulted", "located_lat", "located_lon", "located_match",
    "point_lat", "point_lon", "point_source", "unit_type", "unit_id", "unit_name", "suggestion_rank", "model_pick_rank",
    "suggestion_run", "summary_cut_off", "skip_reason", "notes", "labeler", "labeled_at",
]

# words that describe a place rather than name it
GENERIC = r"\b(villages?|areas?|forests?|jungles?|hills?|police station|ps|block|district|mandal|tehsil|taluka|panchayat|camp|road|near|under|of|the|and)\b"

# other names for SHRUG places, so the names in reports find them (SHRUG name -> names used in reports)
ALIASES = {
    "GHMC": ["Hyderabad", "Greater Hyderabad"],
}

st.set_page_config(page_title="Location labeling", layout="wide")


# ---------------------------------------------------------------- data

def norm(x):
    return re.sub(r"[^a-z]", "", str(x).lower())


@st.cache_data
def load_sample():
    return pd.read_csv(SAMPLE_FILE, dtype=str).fillna("")


@st.cache_data
def load_places():
    places = pd.read_parquet(PLACES_FILE)
    places["name"] = places["name"].fillna("(unnamed)")  # a few SHRUG places have no name
    places["key"] = places["name"].map(norm).where(places["name"] != "(unnamed)", "")
    also = places["name"].map(lambda n: f" [{', '.join(ALIASES[n])}]" if n in ALIASES else "")
    where = ("in " + places["subdistrict_name"].str.title() + " subdistrict").fillna(
        "in " + places["district_name"].str.title() + " district")
    places["label"] = places["name"] + also + " (" + places["place_type"] + ") · " + where
    return places


@st.cache_data
def district_places(state_id, district_id):
    places = load_places()
    return places[(places.state_id == state_id) & (places.district_id == district_id)].reset_index(drop=True)


@st.cache_data
def load_satp():
    """SATP entries for each linked incident: {incident_number: (texts, page url)}."""
    if not (LINKS_FILE.exists() and ENTRIES_FILE.exists()):
        return {}
    entries = pd.read_csv(ENTRIES_FILE, dtype=str)
    text = dict(zip(zip(entries["month"], entries["entry_id"]), entries["text"]))
    links = pd.read_csv(LINKS_FILE, dtype=str).dropna(subset=["entry_id"]).fillna("")
    satp = {}
    for r in links.itertuples():
        keys = [(r.entry_month, r.entry_id)] + [tuple(a.split()) for a in r.also.split(";") if a]
        satp[r.incident_number] = ([text[k] for k in keys], r.url)
    return satp


@st.cache_data
def load_suggestions():
    """Precomputed suggestions and the places extracted from each text (empty until suggest.py has run)."""
    if not (SUGGEST_FILE.exists() and MENTIONS_FILE.exists()):
        return pd.DataFrame(columns=["incident_number", "rank", "model_pick"]), pd.DataFrame(columns=["incident_number", "matched", "role", "name"])
    sug = pd.read_csv(SUGGEST_FILE, dtype={"incident_number": str, "shrid2": str, "subdistrict_id": str})
    if "osm_id" not in sug:  # runs before OpenStreetMap IDs were kept
        sug["osm_id"] = ""
    for c in ("label", "why", "source_note"):  # plain wording; runs before it was added fall back to the evidence
        if c not in sug:
            sug[c] = ""
    text_cols = ["shrid2", "subdistrict_id", "query", "evidence", "model_reason", "name", "relation", "source", "osm_id",
                 "label", "why", "source_note"]
    sug[text_cols] = sug[text_cols].fillna("")
    sug["model_pick"] = sug["model_pick"].astype(str) == "True"
    mentions = pd.read_csv(MENTIONS_FILE, dtype=str).fillna("")
    return sug, mentions


def osm_map(name, osm_id="", lat=None, lon=None):
    """OpenStreetMap: the mapped object itself (outlined), else the spot, else a search for the name in the district."""
    if osm_id:
        return f"https://www.openstreetmap.org/{osm_id}"
    if pd.notna(lat) and pd.notna(lon):
        return f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=14/{lat}/{lon}"
    return "https://www.openstreetmap.org/search?query=" + quote_plus(f"{name}, {inc['district']}, {inc['state']}")


def google_maps(name):
    """A Google Maps search for a place name in the incident's district, filled in."""
    return "https://www.google.com/maps/search/" + quote_plus(f"{name}, {inc['district']}, {inc['state']}, India")


def apply_suggestion(o):
    """Fill the label from a suggestion: the place (or OpenStreetMap point), in or near, or the subdistrict."""
    st.session_state[key("suggestion")] = int(o.rank)
    st.session_state.pop(key("precision"), None)  # the radio takes its value from prec_pick on the next run
    st.session_state[key("via")] = sorted(set(st.session_state.get(key("via"), [])) | {"suggestion"}, key=FOUND_VIA.index)
    if o.source != "OpenStreetMap" and st.session_state.get(key("located_source")) == "OpenStreetMap":
        for k in ("coords_done", "located", "located_shrid", "located_source"):  # a point set by an earlier suggestion
            st.session_state.pop(key(k), None)
        st.session_state[key("coords")] = ""
    if o.relation == "subdistrict":
        st.session_state[key("prec_pick")] = "subdistrict"
        st.session_state[key("sub_pick")] = o.subdistrict_id
        st.session_state.pop(key("subdistrict"), None)
    elif o.relation == "area":  # a point inside a forest, sanctuary, or hill
        st.session_state[key("prec_pick")] = AREA
        st.session_state[key("choice")] = ""
        st.session_state[key("coords")] = st.session_state[key("coords_done")] = f"{o.lat}, {o.lon}"
        shown_as = o.label or f"A point in {o.name}"
        st.session_state[key("located")] = (o.lat, o.lon, shown_as[0].lower() + shown_as[1:] + " (OpenStreetMap)")
        st.session_state[key("located_shrid")] = ""
        st.session_state[key("located_source")] = "OpenStreetMap"
        st.session_state[key("area_name")] = o.name  # saved as the place name
        st.session_state[key("area_label")] = shown_as
        st.session_state[key("type")] = "hill" if re.search(r"OpenStreetMap (peak|hill|ridge)", o.evidence) else "forest"
    else:
        st.session_state[key("prec_pick")] = AT_PLACE[0] if o.relation == "in" else AT_PLACE[1]
        st.session_state[key("choice")] = o.shrid2
        st.session_state[key("type")] = "town" if re.search(r"(SHRUG|OpenStreetMap) (town|city)", o.evidence) else "village"
        if o.source == "OpenStreetMap":
            coords = f"{o.lat}, {o.lon}"
            st.session_state[key("coords")] = st.session_state[key("coords_done")] = coords
            st.session_state[key("located")] = (o.lat, o.lon, "an OpenStreetMap point")
            st.session_state[key("located_shrid")] = o.shrid2
            st.session_state[key("located_source")] = "OpenStreetMap"
    st.rerun()


def load_labels():
    if LABELS_FILE.exists():
        # reindex so files saved before a column was added still load
        return pd.read_csv(LABELS_FILE, dtype=str).reindex(columns=LABEL_COLUMNS).fillna("")
    return pd.DataFrame(columns=LABEL_COLUMNS)


def save_label(row):
    LABELS_FILE.parent.mkdir(exist_ok=True)
    labels = load_labels()
    labels = labels[labels.incident_number != row["incident_number"]]
    labels = pd.concat([labels, pd.DataFrame([row])], ignore_index=True)
    labels[LABEL_COLUMNS].to_csv(LABELS_FILE, index=False)


# ---------------------------------------------------------------- matching

def query_names(inc):
    """Place names the coders recorded, split into single names."""
    names = []
    for field in ("village_name", "other_locations"):
        for part in re.split(r"[,;/&]| and ", inc[field]):
            part = re.sub(GENERIC, " ", part, flags=re.I).strip(" .-()")
            if len(norm(part)) >= 3:
                names.append(part)
    return list(dict.fromkeys(names))


def spelling(a, b, **kwargs):
    """Spelling similarity, 0-1: the average of Jaro-Winkler, which rewards a shared beginning, and an
    edit-distance ratio, which tolerates vowel changes (Revulakota vs. Ravulakota)."""
    return (JaroWinkler.normalized_similarity(a, b) + fuzz.ratio(a, b) / 100) / 2


def shortlist(places, names, n=12):
    """Best SHRUG matches for the coded names, by spelling similarity."""
    keys = places["key"]
    alias_keys = [(i, norm(a)) for i, name in places["name"].items() for a in ALIASES.get(name, [])]
    if alias_keys:
        keys = pd.concat([keys, pd.Series([k for _, k in alias_keys], index=[i for i, _ in alias_keys])])
    # each word of a multi-word name, so "Bacheli" finds "Bade Bacheli"; scored just below an exact match
    parts = pd.Series({(i, w): w for i, name in places["name"].items() if len(str(name).split()) > 1
                       for w in {norm(x) for x in re.split(r"[\s\-()/,]+", str(name))} if len(w) >= 4})
    rows = []
    for name in names:
        for key, score, i in process.extract(norm(name), keys, scorer=spelling, limit=n):
            rows.append((i, name, score))
        for key, score, (i, _) in process.extract(norm(name), parts, scorer=spelling, limit=n) if len(parts) else []:
            if score >= 0.95:
                rows.append((i, name, min(score, 0.96)))
    if not rows:
        return pd.DataFrame(columns=list(places.columns) + ["query", "score"])
    hits = pd.DataFrame(rows, columns=["i", "query", "score"]).sort_values("score", ascending=False).drop_duplicates("i")
    return places.loc[hits.i].assign(query=hits["query"].values, score=hits["score"].values).head(n)


def names_in_text(places, text):
    """SHRUG names in the district that appear as whole words in the summary."""
    words = set(norm(w) for w in re.findall(r"[A-Za-z]+", text))
    pairs = set()
    tokens = re.findall(r"[A-Za-z]+", text)
    for a, b in zip(tokens, tokens[1:]):
        pairs.add(norm(a + b))
    found = places[places.key.str.len().ge(4) & (places.key.isin(words) | places.key.isin(pairs))]
    return found


def split_towns(places):
    """Towns that span several subdistricts, which the Census and SHRUG list as one part per subdistrict (e.g., GHMC)."""
    counts = places[places.place_type == "town"].groupby("name").size()
    return counts[counts > 1].to_dict()


def collapse_towns(df, groups):
    """Show a split town as one candidate, keyed 'town:<name>'; its parts stay available in the search."""
    split = df["name"].isin(groups) & (df["place_type"] == "town")
    if not split.any():
        return df
    whole = df[split].groupby("name", as_index=False).agg(
        lon=("lon", "mean"), lat=("lat", "mean"), query=("query", "first"), score=("score", "max"))
    whole["shrid2"] = "town:" + whole["name"]
    whole["place_type"] = "town"
    whole["label"] = whole["name"].map(lambda n: f"{n}{' [' + ', '.join(ALIASES[n]) + ']' if n in ALIASES else ''} (town) · whole town, {groups[n]} subdistrict parts")
    return pd.concat([whole, df[~split]]).sort_values("score", ascending=False, kind="stable")


def members(pool, choice):
    """SHRUG rows for a choice: one place, or every part of a town that spans subdistricts."""
    if choice.startswith("town:"):
        return pool[(pool["name"] == choice[5:]) & (pool["place_type"] == "town")]
    return pool[pool["shrid2"] == choice]


@st.cache_resource
def load_areas():
    """Outlines of OpenStreetMap forests and protected areas (build_osm.py), by OSM ID."""
    if not AREAS_FILE.exists():
        return {}
    a = pd.read_parquet(AREAS_FILE)
    return dict(zip(a["osm_id"], a["wkb"]))


@st.cache_resource
def state_shapes(state_id):
    """SHRUG boundaries for one state, with a spatial index."""
    shapes = pd.read_parquet(SHAPES_FILE, filters=[("state_id", "==", state_id)])
    geoms = shapely.from_wkb(shapes["wkb"].values)
    return shapes["shrid2"].tolist(), geoms, shapely.STRtree(geoms)


def parse_coords(text):
    """'17.326, 78.555' as copied from Google Maps -> (lat, lon); accepts lon, lat too."""
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if len(nums) != 2:
        return None
    a, b = nums
    if 6 <= a <= 38 and 68 <= b <= 98:
        return a, b
    if 6 <= b <= 38 and 68 <= a <= 98:
        return b, a
    return None


def locate(state_id, lat, lon):
    """The SHRUG place containing a point, or the nearest one within about 2 km."""
    ids, geoms, tree = state_shapes(state_id)
    point = shapely.Point(lon, lat)
    inside = tree.query(point, predicate="intersects")
    if len(inside):
        km2 = shapely.area(geoms[inside[0]]) * 111.32 ** 2 * math.cos(math.radians(lat))
        if km2 > 30:  # top 1% of SHRUG polygons; often an approximation that absorbed forest land
            return ids[inside[0]], (f"inside, but this polygon is unusually large ({km2:.0f} km²; the median village is 2 km²), "
                                    "so the settlement may belong to a nearby village with the matching name")
        return ids[inside[0]], "inside"
    nearest, dist = tree.query_nearest(point, max_distance=0.02, return_distance=True)
    if len(nearest):
        return ids[nearest[0]], f"nearest, about {dist[0] * 111:.1f} km away"
    return None, "no SHRUG place within about 2 km"


@st.cache_resource
def state_subdistricts(state_id):
    """SHRUG subdistrict boundaries for one state, with a spatial index."""
    shapes = pd.read_parquet(SUBDIST_FILE, filters=[("state_id", "==", state_id)])
    geoms = shapely.from_wkb(shapes["wkb"].values)
    return shapes[["district_id", "subdistrict_id"]].to_dict("records"), shapely.STRtree(geoms)


@st.cache_data
def subdistrict_point(state_id, district_id, subdistrict_id):
    """A point to show a subdistrict on the map: its centroid, or a point inside it when the centroid falls outside."""
    shapes = pd.read_parquet(SUBDIST_FILE, filters=[("state_id", "==", state_id)])
    shape = shapes[(shapes.district_id == district_id) & (shapes.subdistrict_id == subdistrict_id)]
    if shape.empty:
        return None
    geom = shapely.from_wkb(shape["wkb"].iloc[0])
    c = shapely.centroid(geom)
    c = c if geom.contains(c) else shapely.point_on_surface(geom)
    return c.y, c.x


def unit_at(state_id, lat, lon):
    """The unit a point falls in: the village or town (shrid2), else the subdistrict (district_id, subdistrict_id)."""
    point = shapely.Point(lon, lat)
    ids, _, tree = state_shapes(state_id)
    inside = tree.query(point, predicate="intersects")
    if len(inside):
        return ids[inside[0]], None
    subs, tree = state_subdistricts(state_id)
    inside = tree.query(point, predicate="intersects")
    return None, (subs[inside[0]] if len(inside) else None)


def preselect(inc, places):
    """An easy case: one exact-name match in the district, or the pin inside a place with the coded name."""
    coded = norm(re.sub(GENERIC, " ", inc["village_name"], flags=re.I))
    if coded:
        exact = places[places.key == coded]
        if len(exact) == 1:
            return exact.iloc[0]["shrid2"], "exact name"
        if inc["pin_place_id"] and JaroWinkler.normalized_similarity(coded, norm(inc["pin_place_name"])) >= 0.9:
            shrid = "-".join(["11", inc["pin_state_id"], inc["pin_district_id"], inc["pin_subdistrict_id"], inc["pin_place_id"]])
            if shrid in set(places.shrid2):
                return shrid, "pin and name"
    return "", ""


def coder_recorded(name, coded):
    """Whether a place name in the text is one the coders recorded."""
    k = norm(name)
    return any(k in norm(c) or norm(c) in k for c in coded if norm(c))


def highlight(text, coded, shrug_names):
    """Yellow: names the coders recorded. Green: other SHRUG places in the text, which the coders did not record."""
    out = html.escape(text)
    for name in sorted({n for n in shrug_names if not coder_recorded(n, coded)}, key=len, reverse=True):
        out = re.sub(rf"\b({re.escape(html.escape(name))})\b", r"<mark style='background:#c6f6d5'>\1</mark>", out, flags=re.I)
    for name in sorted(coded, key=len, reverse=True):
        out = re.sub(rf"({re.escape(html.escape(name))})", r"<mark>\1</mark>", out, flags=re.I)
    return out


def next_unlabeled():
    """After saving: the first unlabeled incident in the sample's order."""
    labeled = set(load_labels().incident_number)
    todo = queue.index[~queue.incident_number.isin(labeled)]
    st.session_state.pos = int(todo[0]) if len(todo) else len(queue) - 1
    st.rerun()


# ---------------------------------------------------------------- page

sample = load_sample()
labels = load_labels()
satp = load_satp()
suggestions, mentions = load_suggestions()
done = set(labels.incident_number)

with st.sidebar:
    st.header("Location labeling")
    labeler = st.text_input("Your initials", key="labeler")
    split = st.radio("Split", ["validation", "test"], horizontal=True)
    queue = sample[sample.split == split].sort_values("order", key=lambda s: s.astype(int)).reset_index(drop=True)
    n_done = queue.incident_number.isin(done).sum()
    st.progress(n_done / len(queue), text=f"{n_done} of {len(queue)} labeled")
    runs = [json.loads(x) for x in RUNS_FILE.read_text().splitlines() if x.strip()] if RUNS_FILE.exists() else []
    runs = [r for r in runs if r["split"] == split]
    if runs:
        r = runs[-1]
        problem = "" if r["status"] == "finished" else f" ⚠️ {r['status']}."
        st.caption(f"Suggestions: run {r['key'][:10]}, {r['incidents']} incidents, {len(r['failed'])} failed, "
                   f"about ${r['estimated_cost_usd']:.2f}.{problem} [OpenAI balance]({BILLING})")
    else:
        st.caption("Suggestions: none yet for this split (run suggest.py).")
    # incidents come in the sample's random order: you can go back to earlier ones, but not past the first unlabeled one
    todo = queue.index[~queue.incident_number.isin(done)]
    frontier = int(todo[0]) if len(todo) else len(queue) - 1
    if "pos" not in st.session_state or st.session_state.get("pos_split") != split:
        st.session_state.pos = frontier
        st.session_state.pos_split = split
    c1, c2 = st.columns(2)
    if c1.button("◀ Previous", use_container_width=True, disabled=st.session_state.pos == 0):
        st.session_state.pos -= 1
    if c2.button("Next ▶", use_container_width=True, disabled=st.session_state.pos >= frontier):
        st.session_state.pos += 1
    if st.button("Back to the next unlabeled", use_container_width=True, disabled=st.session_state.pos == frontier):
        st.session_state.pos = frontier
    with st.expander("Labeling protocol"):
        readme = (ROOT / "README.md").read_text()
        st.markdown(readme.split("## Labeling protocol", 1)[1].split("\n## ", 1)[0])

inc = queue.iloc[st.session_state.pos]
iid = inc["incident_number"]
key = lambda name: f"{name}_{iid}"  # widget keys reset for each incident
existing = labels[labels.incident_number == iid]
existing = existing.iloc[0] if len(existing) else None

places = district_places(inc["state_id"], inc["district_id_2011"]) if inc["district_id_2011"] else load_places().iloc[0:0]
names = query_names(inc)
cands = shortlist(places, names)
satp_texts, satp_page = satp.get(iid, ([], ""))
text = "\n\n".join([inc["incident_summary"]] + satp_texts)  # places in the summary or the full SATP entry
in_text = names_in_text(places, text)
# "Dantewada District" names a district, not the town; keep a name only if it also appears without "District"
in_text = in_text[[bool(re.search(rf"\b{re.escape(n)}\b(?!\s+district)", text, flags=re.I)) for n in in_text["name"]]]
# "Warangal District" names the district, not the town of Warangal
in_text = in_text[~in_text["key"].isin({norm(inc["district"]), norm(inc["district_name_2011"])})]
pre_shrid, pre_source = preselect(inc, places)

if key("choice") not in st.session_state:
    if existing is not None:
        saved = existing["shrid2"]
        st.session_state[key("choice")] = f"town:{existing['place_name']}" if "|" in saved else saved
        if existing["located_lat"]:  # show the saved point without re-running the lookup
            st.session_state[key("coords_done")] = f"{existing['located_lat']}, {existing['located_lon']}"
            st.session_state[key("located")] = (float(existing["located_lat"]), float(existing["located_lon"]), existing["located_match"])
    else:
        st.session_state[key("choice")] = pre_shrid

# header
if existing is not None:
    saved_where = LEGACY.get(existing["precision"], existing["precision"])
    if saved_where == "skipped":
        status = f"⏭ skipped: {existing['skip_reason']}"
    else:
        status = f"✅ labeled: {saved_where}" + (f", {existing['place_name']}" if existing["place_name"] else "")
else:
    status = "not yet labeled"
st.markdown(f"**{split} #{inc['order']}** · incident {iid} · {inc['date']} · {status}")
district_note = inc["district_name_2011"].title() if inc["district_name_2011"] else "no 2011 district matched"
st.markdown(f"### {inc['district']}, {inc['state']}  <span style='font-size:0.7em;color:gray'>2011 district: {district_note}</span>", unsafe_allow_html=True)

left, right = st.columns([3, 2])

with left:
    shrug_in_text = in_text["name"].tolist()
    show = lambda t: st.markdown(f"<div style='font-size:1.1em;line-height:1.6'>{highlight(t, names, shrug_in_text)}</div>",
                                 unsafe_allow_html=True)
    show(inc["incident_summary"])
    if not re.search(r"[.!?\"'”’)\]]$", inc["incident_summary"].strip()):
        st.warning("This summary may be cut off (no final punctuation)" + (": see the full SATP entry." if satp_texts else "."))
    if satp_texts:
        with st.expander("Full SATP entry" + (" (several entries)" if len(satp_texts) > 1 else "") + ": may describe other incidents too"):
            for t in satp_texts:
                show(t)
    else:
        st.caption("Not on SATP's current pages: the coded summary is the only text.")
    st.caption(f"Coded village: **{inc['village_name'] or '—'}** · other locations: **{inc['other_locations'] or '—'}**  "
               "· yellow = names the coders recorded · green = places in the text the coders did not record")
    gq = quote_plus(f"{names[0] if names else ''} {inc['district']} {inc['state']} India")
    st.markdown(f"[SATP page]({satp_page or inc['satp_url']}) · [Google Maps](https://www.google.com/maps/search/{gq})")

    sug = suggestions[suggestions.incident_number == iid]
    if not len(sug) and len(suggestions):
        st.caption("No suggestions for this incident: suggest.py has not reached it, or it failed (see the sidebar).")
    if len(sug):
        st.subheader("Suggestions")
        # highlight the suggestion last chosen, while the label still matches it
        current = st.session_state.get(key("precision")) or st.session_state.get(key("prec_pick"))
        implied = {"in": AT_PLACE[0], "near": AT_PLACE[1], "area": AREA, "subdistrict": "subdistrict"}
        for o in sug[sug["rank"] > 0].itertuples():
            on = (st.session_state.get(key("suggestion")) == o.rank and current == implied.get(o.relation)
                  and (o.relation in ("area", "subdistrict") or st.session_state[key("choice")] == o.shrid2))
            mark = " ★ model's pick" if o.model_pick else ""
            c_opt, c_g, c_o = st.columns([6, 1, 1], vertical_alignment="center")
            title = o.label or f"{o.relation} {o.name} · {o.source}"
            if c_opt.button(f"{'● ' if on else ''}{o.rank}. {title}{mark}", key=key(f"s_{o.rank}"),
                            type="primary" if on else "secondary", use_container_width=True):
                apply_suggestion(o)
            c_g.link_button("Google", google_maps(o.query or o.name), use_container_width=True)
            c_o.link_button("OSM", osm_map(o.name, o.osm_id, o.lat, o.lon), use_container_width=True)
            st.caption(f"**Why:** {o.why}  \n**Source:** {o.source_note}" if o.why else o.evidence)
        if sug[sug.model_pick]["rank"].eq(0).any():
            st.caption("The model picked none of these.")
    # places the text puts the event in or near that the search did not find closely (SHRUG spelling below 0.92,
    # nothing in OpenStreetMap); district and state names are never searched, and a matched subdistrict is found
    weak = mentions[(mentions.incident_number == iid)
                    & mentions.role.isin(["where it happened", "near where it happened"])
                    & ~mentions.kind.isin(["district", "state"])
                    & ~mentions.matched.str.contains("OpenStreetMap|subdistrict")].copy()
    weak["shrug"] = weak.matched.str.extract(r"SHRUG (\d\.\d+)")[0].astype(float).fillna(0)
    weak = weak[weak.shrug < 0.92]
    osm_kinds = ["village", "town", "hamlet", "locality", "police station", "camp", "other", "forest", "hill"]
    g = lambda m: (f"Look up {m.name} on [Google Maps]({google_maps(m.name)}) · "
                   f"[OpenStreetMap]({osm_map(m.name)})")

    for m in weak.itertuples():
        if m.kind in osm_kinds:
            missing = f"**{m.name}** is not in SHRUG or OpenStreetMap" + ("." if m.shrug else " under any similar spelling.")
        else:
            kinds = "mandals or blocks" if m.kind == "mandal or block" else m.kind + "s"
            missing = f"**{m.name}** ({m.kind}) is not in SHRUG, and OpenStreetMap is not searched for {kinds}."
        closest = ""
        if m.shrug:
            opt = sug[(sug["query"] == m.name) & (sug["source"] == "SHRUG") & (sug["rank"] > 0)]
            closest = (f" The closest SHRUG name, {opt.iloc[0]['name']} (option {opt.iloc[0]['rank']}, spelling similarity "
                       f"{m.shrug:.2f}), may be a different place." if len(opt) else
                       f" The closest SHRUG name (spelling similarity {m.shrug:.2f}) is not among the options.")
        st.markdown(f"{missing}{closest} {g(m)}")
    cand_box = st.expander("More candidates: coded names and names in the text") if len(sug) else st.container()
    if not len(sug):
        st.subheader("Candidates")
    with cand_box:
        st.caption("Ranked by how closely the spelling matches a coded name (1.00 = same spelling); this is not a probability.")
        groups = split_towns(places)
        shown = pd.concat([in_text.assign(query="", score=1.0), cands]).drop_duplicates("shrid2")
        shown = collapse_towns(shown, groups)
        for _, p in shown.iterrows():
            is_on = st.session_state[key("choice")] == p["shrid2"]
            if p["query"]:
                why = f"spelling match to “{p['query']}” {p['score']:.2f}"
            else:
                in_summary = re.search(rf"\b{re.escape(p['name'])}\b", inc["incident_summary"], flags=re.I)
                why = ("named in the summary" if in_summary else "named only in the full SATP entry") + (
                    "" if coder_recorded(p["name"], names) else ", not recorded by the coders")
            label = f"{'● ' if is_on else ''}{p['label']} · {why}"
            if st.button(label, key=key(f"c_{p['shrid2']}"), type="primary" if is_on else "secondary"):
                st.session_state[key("choice")] = p["shrid2"]
                st.session_state[key("via_search")] = False
                st.rerun()
            if p["shrid2"].startswith("town:"):  # the town's subdistrict parts, indented under it
                parts = members(places, p["shrid2"]).sort_values("subdistrict_name", na_position="last")
                _, inner = st.columns([1, 12])
                for _, part in parts.iterrows():
                    part_on = st.session_state[key("choice")] == part["shrid2"]
                    sub = part["subdistrict_name"].title() if isinstance(part["subdistrict_name"], str) else "unnamed subdistrict"
                    if inner.button(f"{'● ' if part_on else ''}↳ part in {sub}", key=key(f"c_{part['shrid2']}"),
                                    type="primary" if part_on else "secondary"):
                        st.session_state[key("choice")] = part["shrid2"]
                        st.rerun()

    lookup = st.text_input("Look up a name", key=key("lookup"), placeholder="e.g. a village named on the SATP page",
                           help="Lists the closest spellings among SHRUG villages and towns in the district. A close spelling "
                                "may or may not be the same place: check on a map before choosing.")
    if lookup.strip():
        for _, p in shortlist(places, [lookup.strip()], n=8).iterrows():
            is_on = st.session_state[key("choice")] == p["shrid2"]
            if st.button(f"{'● ' if is_on else ''}{p['label']} · spelling match {p['score']:.2f}",
                         key=key(f"l_{p['shrid2']}"), type="primary" if is_on else "secondary"):
                st.session_state[key("choice")] = p["shrid2"]
                st.rerun()

    widen = st.checkbox("Search other districts in this state too", key=key("widen"),
                        help="For events the coders placed in the wrong district, or places in a neighboring district")
    all_places = load_places()
    pool = all_places[all_places.state_id == inc["state_id"]] if widen else places
    if (choice_now := st.session_state[key("choice")]) and not choice_now.startswith("town:"):
        pool = pd.concat([pool, all_places[all_places.shrid2 == choice_now]]).drop_duplicates("shrid2")
    whole_towns = {f"town:{n}": f"{n} (town) · whole town, {k} subdistrict parts" for n, k in groups.items()}
    options = [""] + list(whole_towns) + pool["shrid2"].tolist()
    labels_by_shrid = dict(zip(pool["shrid2"], pool["label"] + ("" if not widen else ", " + pool["district_name"].str.title() + " district")))
    labels_by_shrid.update(whole_towns)
    picked = st.selectbox("Search villages and towns" + (" in the state" if widen else " in the district"), options,
                          key=key("search"), format_func=lambda s: labels_by_shrid.get(s, "type to search…"))
    if picked and picked != st.session_state[key("choice")]:
        st.session_state[key("choice")] = picked
        st.session_state[key("via_search")] = True
        st.rerun()

    saved_coords = f"{existing['located_lat']}, {existing['located_lon']}" if existing is not None and existing["located_lat"] else ""
    def clear_point(reset_box=True):
        """Forget the pasted point, and the place it selected."""
        if st.session_state.get(key("choice")) == st.session_state.get(key("located_shrid")):
            st.session_state[key("choice")] = ""
        for k in ("coords_done", "located", "located_shrid", "located_source"):
            st.session_state.pop(key(k), None)
        if reset_box:  # only from the button's callback, before the box is drawn
            st.session_state[key("coords")] = ""

    if key("coords") not in st.session_state:
        st.session_state[key("coords")] = saved_coords
    c_in, c_clear = st.columns([5, 1], vertical_alignment="bottom")
    coords = c_in.text_input("Locate by coordinates", key=key("coords"),
                             placeholder="e.g. 17.3260, 78.5550",
                             help="In Google Maps, right-click the spot and click the coordinates at the top of the menu to copy them; paste here.")
    c_clear.button("Clear", key=key("clear_point"), on_click=clear_point, use_container_width=True)
    if not coords and st.session_state.get(key("coords_done")):  # box emptied by hand
        clear_point(reset_box=False)
        st.rerun()
    if coords and coords != st.session_state.get(key("coords_done")):
        st.session_state[key("coords_done")] = coords
        st.session_state.pop(key("located_source"), None)  # a pasted point is yours
        parsed = parse_coords(coords)
        if parsed is None:
            st.session_state[key("located")] = (None, None, "could not read coordinates in India")
        else:
            shrid, how = locate(inc["state_id"], *parsed)
            st.session_state[key("located")] = (*parsed, how)
            st.session_state[key("located_shrid")] = shrid
            if shrid:
                st.session_state[key("choice")] = shrid
            st.rerun()
    if located := st.session_state.get(key("located")):
        lat, lon, how = located
        here = all_places[all_places.shrid2 == st.session_state[key("choice")]]
        if lat is None:
            st.warning(f"Locate: {how}.")
        elif here.empty:
            st.info(f"Point: {how}.")
        else:
            row = here.iloc[0]
            note = "" if row["district_id"] == inc["district_id_2011"] else f" (in {row['district_name'].title()} district, not the coded one)"
            st.success(f"Point is {how}: {row['label']}{note}")

with right:
    if inc["latitude"]:
        show_pin = st.toggle("Show the India team's pin", key=key("show_pin"))
        if show_pin:
            st.session_state[key("pin_consulted")] = True
    else:
        show_pin = False
        st.caption("No pin: the India team entered no coordinates for this incident.")
    choice = st.session_state[key("choice")]
    # the suggestions as numbered pins; the shortlist of names like the coders' fainter behind them
    sug_pts = sug[(sug["rank"] > 0) & sug["lat"].notna()] if len(sug) else sug.iloc[0:0]
    sug_pts = pd.DataFrame({"shrid2": sug_pts["shrid2"], "name": sug_pts["rank"].astype(int).astype(str) + ". " + sug_pts["name"],
                            "lon": sug_pts["lon"].astype(float), "lat": sug_pts["lat"].astype(float), "kind": "suggestion"})
    pts = shown[["shrid2", "name", "lon", "lat"]].assign(kind="candidate")
    pts = pts[~pts.shrid2.isin(set(sug_pts.shrid2) - {""})]
    pts = pd.concat([pts, sug_pts])
    if choice and choice not in set(pts.shrid2):
        pts = pd.concat([pts, members(pool, choice)[["shrid2", "name", "lon", "lat"]].assign(kind="candidate")])
    pts.loc[(pts.shrid2 == choice) & (pts.shrid2 != ""), "kind"] = "selected"
    if (located := st.session_state.get(key("located"))) and located[0] is not None:
        pts = pd.concat([pts, pd.DataFrame([{"shrid2": "", "name": "your point", "lon": located[1], "lat": located[0], "kind": "located"}])])
    if show_pin and inc["latitude"]:
        pts = pd.concat([pts, pd.DataFrame([{"shrid2": "", "name": "pin", "lon": float(inc["longitude"]), "lat": float(inc["latitude"]), "kind": "pin"}])])
    colors = {"candidate": [43, 108, 176, 90], "suggestion": [106, 61, 154, 230], "selected": [221, 107, 32, 230],
              "pin": [200, 30, 30, 230], "located": [34, 139, 34, 230]}
    # outlines of the forests and sanctuaries among the suggestions (protocol rule 4)
    outlines = load_areas()
    area_sug = sug[(sug["relation"] == "area") & sug["osm_id"].isin(outlines)].drop_duplicates("osm_id") if len(sug) else sug
    features = [{"type": "Feature", "properties": {"name": o.name.split(",")[0]},
                 "geometry": json.loads(shapely.to_geojson(shapely.from_wkb(outlines[o.osm_id])))} for o in area_sug.itertuples()]
    map_slot = st.container()  # drawn once the label is known, so a chosen subdistrict can be shown

    def draw_map(pts):
        pts = pts.assign(color=pts["kind"].map(colors),
                         text_color=[[120, 120, 120, 200] if k == "candidate" else [20, 20, 20, 255] for k in pts["kind"]])
        # fit the view to what matters: the suggestions, the selected place, your point, and the pin when shown
        focus = pts[pts["kind"] != "candidate"]
        focus = focus if len(focus) else pts
        if len(focus):
            lat0, lat1, lon0, lon1 = focus.lat.min(), focus.lat.max(), focus.lon.min(), focus.lon.max()
            span = max(lat1 - lat0, (lon1 - lon0) * math.cos(math.radians((lat0 + lat1) / 2)), 0.03)
            view = pdk.ViewState(latitude=(lat0 + lat1) / 2, longitude=(lon0 + lon1) / 2,
                                 zoom=max(4, min(11, math.log2(390 / span))))  # 420 px tall map, with a margin
        else:
            view = pdk.ViewState(latitude=21, longitude=82, zoom=4)
        map_slot.pydeck_chart(pdk.Deck(
            layers=[
                pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": features}, stroked=True, filled=True,
                          get_fill_color=[34, 139, 34, 40], get_line_color=[34, 110, 34, 220], line_width_min_pixels=2, pickable=True),
                pdk.Layer("ScatterplotLayer", pts, get_position="[lon, lat]", get_fill_color="color", get_radius=600, pickable=True),
                pdk.Layer("TextLayer", pts, get_position="[lon, lat]", get_text="name", get_size=13, get_color="text_color",
                          get_pixel_offset=[0, -14]),
            ],
            initial_view_state=view, tooltip={"text": "{name}"}, map_style=None,
        ), height=420)
        map_slot.caption("Purple: suggestions (numbered) · orange: selected (for a subdistrict, its center) · green: your point "
                         "and forest outlines · red: the India team's pin · faint blue: other places named like the coders' entries")

    if pre_shrid:
        st.info(f"Pre-selected ({pre_source}): {labels_by_shrid.get(pre_shrid, pre_shrid)}. Confirm or change it.")

    st.subheader("Label")
    default = lambda col, fallback: existing[col] if existing is not None and existing[col] else fallback
    if existing is not None and existing["precision"] == "skipped":
        default = lambda col, fallback: existing[col] if col in ("notes",) and existing[col] else fallback
    saved_named = default("named_below_district", "")
    named = st.radio("Does the report name a place below the district for where it happened?", NAMED, key=key("named"),
                     index=NAMED.index(saved_named) if saved_named in NAMED else None, horizontal=True,
                     help="Any village, town, police station, camp, forest, hill, road, or subdistrict that locates the event, "
                          "whether or not you can find it. Not where victims lived or where forces came from (rule 1).")
    # blank until a suggestion is clicked or a label is saved, so the labeler always chooses
    start_where = st.session_state.get(key("prec_pick")) or LEGACY.get(default("precision", ""), default("precision", ""))
    precision = st.radio("Where did it happen?", PRECISIONS, key=key("precision"),
                         index=PRECISIONS.index(start_where) if start_where in PRECISIONS else None, horizontal=True)
    chosen = members(pool, choice) if choice else pool.iloc[0:0]
    subdistrict_id = chosen["subdistrict_id"].iloc[0] if chosen["subdistrict_id"].nunique() == 1 else ""
    # the district the label falls in: the selected place's, which can differ from the coded one
    district_id = chosen["district_id"].iloc[0] if len(chosen) else inc["district_id_2011"]
    in_district = all_places[(all_places.state_id == inc["state_id"]) & (all_places.district_id == district_id)]
    if precision in AT_PLACE:
        st.markdown(f"**Selected:** {labels_by_shrid.get(choice, '— none: pick a candidate or search —')}")
    elif precision == AREA:
        st.markdown(f"**Point:** {st.session_state.get(key('area_label')) or default('place_name', '') or '— pick a forest suggestion, or paste a point in the forest —'}")
    elif precision == "subdistrict":
        # the coded district's subdistricts first, then the selected place's district if it differs
        coded_district = all_places[(all_places.state_id == inc["state_id"]) & (all_places.district_id == inc["district_id_2011"])]
        subs = pd.concat([coded_district, in_district]).drop_duplicates("subdistrict_id")
        subs = subs.assign(other=subs.district_id != inc["district_id_2011"]).sort_values(["other", "subdistrict_name"])
        sub_names = dict(zip(subs.subdistrict_id, subs.subdistrict_name.str.title()
                             + subs.other.map({True: " (", False: ""}) + subs.district_name.str.title().where(subs.other, "")
                             + subs.other.map({True: " district)", False: ""})))
        sub_district = dict(zip(subs.subdistrict_id, subs.district_id))
        options = [""] + subs.subdistrict_id.tolist()
        saved = existing["subdistrict_id"] if existing is not None and existing["precision"] == "subdistrict" else ""
        # default: the selected village's subdistrict, else one the coders named (e.g., "Thullur" mandal)
        coder_subs = [i for i, n in sub_names.items() if isinstance(n, str) and norm(n) in {norm(x) for x in names}]
        start = st.session_state.get(key("sub_pick")) or saved or subdistrict_id or (coder_subs[0] if coder_subs else "")
        subdistrict_id = st.selectbox("Subdistrict (defaults to the selected village's)", options, key=key("subdistrict"),
                                      index=options.index(start) if start in options else 0,
                                      format_func=lambda s: sub_names.get(s, "choose…"))
        if subdistrict_id:  # the label's district follows the chosen subdistrict
            district_id = sub_district[subdistrict_id]
            in_district = all_places[(all_places.state_id == inc["state_id"]) & (all_places.district_id == district_id)]
    # suggestions set these two, so they take their starting values from session state, not index or default
    st.session_state.setdefault(key("type"), default("place_type", "village"))
    st.session_state.setdefault(key("via"), [v for v in default("found_via", "").split("|") if v])
    place_type = st.selectbox("What kind of place is it?", PLACE_TYPES, key=key("type"))
    reason = st.selectbox("If not a village or town: why?", REASONS, key=key("reason"),
                          index=REASONS.index(default("reason", "")))
    found_via = st.multiselect("How did you find it?", FOUND_VIA, key=key("via"))
    saved_certainty = default("certainty", "")
    certainty = st.radio("How sure are you that this is the right place?", CERTAINTY, key=key("certainty"),
                         index=CERTAINTY.index(saved_certainty) if saved_certainty in CERTAINTY else None, horizontal=True,
                         help="Apart from how large the area is. Sure: one clear match. Probable: the best of several "
                              "namesakes, a spelling stretch, or a reading of the text you had to infer. Guess: a plausible "
                              "place on weak evidence. Say why in the notes unless sure.")
    cut_off = st.checkbox("Summary is cut off (labeled from the SATP page)", key=key("cut_off"),
                          value=default("summary_cut_off", "False") == "True")
    notes = st.text_input("Notes", key=key("notes"), value=default("notes", ""))

    # point for mapping and unit for analysis (protocol rule 6)
    loc = st.session_state.get(key("located"))
    point, point_source = None, ""
    if loc and loc[0] is not None:
        point, point_source = (loc[0], loc[1]), st.session_state.get(key("located_source"), "your point")
    elif precision in AT_PLACE and len(chosen):
        point, point_source = (chosen["lat"].mean(), chosen["lon"].mean()), "SHRUG place"
    sub_lookup = all_places.drop_duplicates(["district_id", "subdistrict_id"]).set_index(["district_id", "subdistrict_id"])["subdistrict_name"]
    unit_type = unit_id = unit_name = ""
    if precision in AT_PLACE and point_source == "SHRUG place":
        unit_type, unit_id, unit_name = chosen["place_type"].iloc[0], "|".join(chosen["shrid2"]), chosen["name"].iloc[0]
    elif (precision in AT_PLACE or precision == AREA) and point:
        in_place, in_sub = unit_at(inc["state_id"], *point)
        if in_place:
            here = all_places[all_places.shrid2 == in_place]
            unit_type, unit_id = "village or town", in_place
            if len(here):
                unit_type, unit_name = here["place_type"].iloc[0], here["name"].iloc[0]
            if precision in AT_PLACE and len(chosen) and in_place not in set(chosen["shrid2"]):
                st.warning(f"Your point falls in {unit_name or in_place}, not the selected place; the unit follows the point.")
        elif in_sub:
            unit_type, unit_id = "subdistrict", f"{inc['state_id']}-{in_sub['district_id']}-{in_sub['subdistrict_id']}"
            unit_name = str(sub_lookup.get((in_sub["district_id"], in_sub["subdistrict_id"]), "")).title()
    elif precision == "subdistrict" and subdistrict_id:
        unit_type, unit_id = "subdistrict", f"{inc['state_id']}-{district_id}-{subdistrict_id}"
        unit_name = str(sub_lookup.get((district_id, subdistrict_id), "")).title()
    elif precision == "district":
        unit_type, unit_id = "district", f"{inc['state_id']}-{district_id}"
        unit_name = in_district["district_name"].iloc[0].title() if len(in_district) else ""
    elif precision == "state or unknown":
        unit_type, unit_id = "state", inc["state_id"]
    if unit_type:
        where = f"{point[0]:.4f}, {point[1]:.4f} ({point_source})" if point else "none"
        st.caption(f"Saved as: point {where} · unit for analysis: {unit_name or unit_id} ({unit_type})"
                   + (" · the point is outside every village" if unit_type == "subdistrict" and point else ""))

    if precision == "subdistrict" and subdistrict_id and (c := subdistrict_point(inc["state_id"], district_id, subdistrict_id)):
        pts = pd.concat([pts, pd.DataFrame([{"shrid2": "", "name": f"{unit_name} subdistrict", "lon": c[1], "lat": c[0], "kind": "selected"}])])
    draw_map(pts)

    # which suggestion the label matches (rank 1-5, or "not in list"), and the model's pick (0 = none)
    suggestion_rank, model_pick_rank = "", ""
    if len(sug):
        model_pick_rank = str(int(sug[sug.model_pick]["rank"].iloc[0])) if sug.model_pick.any() else ""
        suggestion_rank = "not in list"
        for o in sug[sug["rank"] > 0].itertuples():
            if o.relation == "subdistrict":
                hit = precision == "subdistrict" and o.subdistrict_id == subdistrict_id
            elif o.relation == "area":
                hit = precision == AREA and point and abs(point[0] - o.lat) < 1e-5 and abs(point[1] - o.lon) < 1e-5
            elif o.source == "OpenStreetMap":
                hit = precision in AT_PLACE and point_source == "OpenStreetMap" and point and abs(point[0] - o.lat) < 1e-5
            else:
                hit = precision in AT_PLACE and o.shrid2 in set(chosen["shrid2"])
            if hit:
                suggestion_rank = str(int(o.rank))
                break

    if st.button("Save and go to next", type="primary", use_container_width=True):
        if not labeler:
            st.error("Enter your initials in the sidebar first.")
        elif named is None:
            st.error("Say whether the report names a place below the district.")
        elif precision is None:
            st.error("Say where it happened.")
        elif named == "no" and precision not in ("district", "state or unknown"):
            st.error("Nothing named below the district: label the district (or the state).")
        elif certainty is None:
            st.error("Say how sure you are that this is the right place.")
        elif certainty != "sure" and not notes.strip():
            st.error("Say in the notes what you were unsure about.")
        elif precision in AT_PLACE and not choice:
            st.error("Pick a village or town, or choose a coarser unit.")
        elif precision == "subdistrict" and not subdistrict_id:
            st.error("Choose the subdistrict.")
        elif precision == AREA and not point:
            st.error("Pick a forest suggestion, or paste a point in the forest under Locate by coordinates.")
        else:
            row = chosen.iloc[0] if precision in AT_PLACE else None
            if precision in ("district", "state or unknown"):
                subdistrict_id = ""
            if precision == AREA:  # the subdistrict and district come from the point's unit
                if unit_type == "subdistrict":
                    _, district_id, subdistrict_id = unit_id.split("-")
                elif len(here := all_places[all_places.shrid2 == unit_id]):
                    district_id, subdistrict_id = here["district_id"].iloc[0], here["subdistrict_id"].iloc[0]
                in_district = all_places[(all_places.state_id == inc["state_id"]) & (all_places.district_id == district_id)]
            sub_name = in_district.loc[in_district.subdistrict_id == subdistrict_id, "subdistrict_name"]
            save_label({
                "incident_number": iid, "split": split, "order": inc["order"], "named_below_district": named,
                "precision": precision, "certainty": certainty,
                "shrid2": "|".join(chosen["shrid2"]) if row is not None else "",  # several IDs for a whole split town
                "place_name": row["name"] if row is not None else
                              (st.session_state.get(key("area_name")) or default("place_name", "")) if precision == AREA else "",
                "subdistrict_id": subdistrict_id,
                "subdistrict_name": sub_name.iloc[0] if subdistrict_id and len(sub_name) else "",
                "state_id": inc["state_id"], "district_id_2011": district_id if precision != "state or unknown" else "",
                "place_type": place_type, "reason": reason, "found_via": "|".join(found_via),
                "preselected": pre_shrid, "preselect_source": pre_source,
                "confirmed": str(bool(pre_shrid) and choice == pre_shrid and precision == AT_PLACE[0]),
                "pin_consulted": str(st.session_state.get(key("pin_consulted"), False)),
                "located_lat": loc[0] if (loc := st.session_state.get(key("located"))) and loc[0] is not None else "",
                "located_lon": loc[1] if loc and loc[0] is not None else "",
                "located_match": loc[2] if loc and loc[0] is not None else "",
                "point_lat": f"{point[0]:.6f}" if point else "", "point_lon": f"{point[1]:.6f}" if point else "",
                "point_source": point_source, "unit_type": unit_type, "unit_id": unit_id, "unit_name": unit_name,
                "suggestion_rank": suggestion_rank, "model_pick_rank": model_pick_rank,
                "suggestion_run": sug["run"].iloc[0] if len(sug) and "run" in sug else "",  # which suggest.py run was shown
                "summary_cut_off": str(cut_off), "notes": notes, "labeler": labeler, "labeled_at": datetime.now().isoformat(timespec="seconds"),
            })
            next_unlabeled()

    with st.expander("Skip this incident"):
        st.caption("Only when it cannot be labeled. Skipped incidents stay in the sample and count in the denominator.")
        skip_reason = st.selectbox("Why?", SKIP_REASONS, key=key("skip_reason"),
                                   index=SKIP_REASONS.index(existing["skip_reason"]) if existing is not None
                                   and existing["skip_reason"] in SKIP_REASONS else 0)
        if st.button("Skip with this reason", use_container_width=True):
            if not labeler:
                st.error("Enter your initials in the sidebar first.")
            elif not skip_reason:
                st.error("Choose a reason.")
            elif skip_reason.startswith("other") and not notes.strip():
                st.error("Say why in the notes.")
            else:
                save_label({
                    "incident_number": iid, "split": split, "order": inc["order"], "precision": "skipped",
                    "state_id": inc["state_id"], "skip_reason": skip_reason, "notes": notes, "labeler": labeler,
                    "labeled_at": datetime.now().isoformat(timespec="seconds"),
                })
                next_unlabeled()
