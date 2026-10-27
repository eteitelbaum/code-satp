"""Location labeling app: assign each sampled incident to a Census 2011 place (SHRUG).

Run from the repository root, after build_lookup.R and draw_sample.py:
    streamlit run location-labeling/app.py

For each incident, record the most precise unit you are confident of: a village or
town (SHRUG ID), otherwise the subdistrict, otherwise the district. Labels autosave
to location-labeling/labels/labels.csv, one row per incident, so you can stop and
resume at any time.
"""

import html
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
LABELS_FILE = ROOT / "labels" / "labels.csv"

PRECISIONS = ["in a village or town", "near a village or town", "subdistrict", "district", "state or unknown"]
AT_PLACE = PRECISIONS[:2]  # precisions that name a SHRUG place
LEGACY = {"village or town": "in a village or town"}  # earlier label values
PLACE_TYPES = ["village", "town", "police station", "camp", "forest", "hill", "river", "road", "other"]
REASONS = ["", "hamlet", "local name", "area only", "camp", "not found", "other"]
FOUND_VIA = ["shortlist", "search", "Google Maps", "SATP page", "pin"]
LABEL_COLUMNS = [
    "incident_number", "split", "order", "precision", "shrid2", "place_name", "subdistrict_id",
    "subdistrict_name", "state_id", "district_id_2011", "place_type", "reason", "found_via",
    "preselected", "preselect_source", "confirmed", "pin_consulted", "located_lat", "located_lon", "located_match",
    "summary_cut_off", "notes", "labeler", "labeled_at",
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
    rows = []
    for name in names:
        for key, score, i in process.extract(norm(name), keys, scorer=spelling, limit=n):
            rows.append((i, name, score))
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


def highlight(text, coded, shrug_names):
    out = html.escape(text)
    for name in sorted(shrug_names, key=len, reverse=True):
        out = re.sub(rf"\b({re.escape(html.escape(name))})\b", r"<span style='border-bottom:2px solid #2b6cb0'>\1</span>", out, flags=re.I)
    for name in sorted(coded, key=len, reverse=True):
        out = re.sub(rf"({re.escape(html.escape(name))})", r"<mark>\1</mark>", out, flags=re.I)
    return out


# ---------------------------------------------------------------- page

sample = load_sample()
labels = load_labels()
done = set(labels.incident_number)

with st.sidebar:
    st.header("Location labeling")
    labeler = st.text_input("Your initials", key="labeler")
    split = st.radio("Split", ["validation", "test"], horizontal=True)
    queue = sample[sample.split == split].sort_values("order", key=lambda s: s.astype(int)).reset_index(drop=True)
    n_done = queue.incident_number.isin(done).sum()
    st.progress(n_done / len(queue), text=f"{n_done} of {len(queue)} labeled")
    if "pos" not in st.session_state or st.session_state.get("pos_split") != split:
        todo = queue.index[~queue.incident_number.isin(done)]
        st.session_state.pos = int(todo[0]) if len(todo) else 0
        st.session_state.pos_split = split
    c1, c2 = st.columns(2)
    if c1.button("◀ Previous", use_container_width=True):
        st.session_state.pos = max(0, st.session_state.pos - 1)
    if c2.button("Next ▶", use_container_width=True):
        st.session_state.pos = min(len(queue) - 1, st.session_state.pos + 1)
    if st.button("Next unlabeled", use_container_width=True):
        todo = queue.index[~queue.incident_number.isin(done) & (queue.index > st.session_state.pos)]
        if len(todo):
            st.session_state.pos = int(todo[0])
    jump = st.number_input("Go to order #", 1, len(queue), st.session_state.pos + 1)
    if st.button("Go", use_container_width=True):
        st.session_state.pos = int(jump) - 1
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
in_text = names_in_text(places, inc["incident_summary"])
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
    status = f"✅ labeled: {saved_where}" + (f", {existing['place_name']}" if existing["place_name"] else "")
else:
    status = "not yet labeled"
st.markdown(f"**{split} #{inc['order']}** · incident {iid} · {inc['date']} · {status}")
district_note = inc["district_name_2011"].title() if inc["district_name_2011"] else "no 2011 district matched"
st.markdown(f"### {inc['district']}, {inc['state']}  <span style='font-size:0.7em;color:gray'>2011 district: {district_note}</span>", unsafe_allow_html=True)

left, right = st.columns([3, 2])

with left:
    st.markdown(
        f"<div style='font-size:1.1em;line-height:1.6'>{highlight(inc['incident_summary'], names, in_text['name'].tolist())}</div>",
        unsafe_allow_html=True,
    )
    if not re.search(r"[.!?\"'”’)\]]$", inc["incident_summary"].strip()):
        st.warning("This summary may be cut off (no final punctuation). Check the SATP source page for the full entry.")
    st.caption(f"Coded village: **{inc['village_name'] or '—'}** · other locations: **{inc['other_locations'] or '—'}**  "
               "· yellow = coded names, underlined = SHRUG names in this district")
    gq = quote_plus(f"{names[0] if names else ''} {inc['district']} {inc['state']} India")
    st.markdown(f"[SATP source page]({inc['satp_url']}) · [Google Maps](https://www.google.com/maps/search/{gq})")

    st.subheader("Candidates")
    st.caption("Ranked by how closely the spelling matches a coded name (1.00 = same spelling); this is not a probability.")
    groups = split_towns(places)
    shown = pd.concat([in_text.assign(query="", score=1.0), cands]).drop_duplicates("shrid2")
    shown = collapse_towns(shown, groups)
    for _, p in shown.iterrows():
        is_on = st.session_state[key("choice")] == p["shrid2"]
        why = "named in the text" if not p["query"] else f"spelling match to “{p['query']}” {p['score']:.2f}"
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
        for k in ("coords_done", "located", "located_shrid"):
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
        if lat is None or here.empty:
            st.warning(f"Locate: {how}.")
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
    pts = shown[["shrid2", "name", "lon", "lat"]].assign(kind="candidate")
    if choice and choice not in set(pts.shrid2):
        pts = pd.concat([pts, members(pool, choice)[["shrid2", "name", "lon", "lat"]].assign(kind="candidate")])
    pts.loc[pts.shrid2 == choice, "kind"] = "selected"
    if (located := st.session_state.get(key("located"))) and located[0] is not None:
        pts = pd.concat([pts, pd.DataFrame([{"shrid2": "", "name": "your point", "lon": located[1], "lat": located[0], "kind": "located"}])])
    if show_pin and inc["latitude"]:
        pts = pd.concat([pts, pd.DataFrame([{"shrid2": "", "name": "pin", "lon": float(inc["longitude"]), "lat": float(inc["latitude"]), "kind": "pin"}])])
    colors = {"candidate": [43, 108, 176, 180], "selected": [221, 107, 32, 230], "pin": [200, 30, 30, 230],
              "located": [34, 139, 34, 230]}
    pts["color"] = pts["kind"].map(colors)
    if len(pts):
        view = pdk.ViewState(latitude=pts.lat.mean(), longitude=pts.lon.mean(), zoom=9)
    else:
        view = pdk.ViewState(latitude=21, longitude=82, zoom=4)
    st.pydeck_chart(pdk.Deck(
        layers=[
            pdk.Layer("ScatterplotLayer", pts, get_position="[lon, lat]", get_fill_color="color", get_radius=600, pickable=True),
            pdk.Layer("TextLayer", pts, get_position="[lon, lat]", get_text="name", get_size=13, get_pixel_offset=[0, -14]),
        ],
        initial_view_state=view, tooltip={"text": "{name}"}, map_style=None,
    ), height=420)
    if pre_shrid:
        st.info(f"Pre-selected ({pre_source}): {labels_by_shrid.get(pre_shrid, pre_shrid)}. Confirm or change it.")

    st.subheader("Label")
    default = lambda col, fallback: existing[col] if existing is not None and existing[col] else fallback
    precision = st.radio("Where did it happen?", PRECISIONS, key=key("precision"),
                         index=PRECISIONS.index(LEGACY.get(default("precision", ""), default("precision", PRECISIONS[0]))),
                         horizontal=True)
    chosen = members(pool, choice) if choice else pool.iloc[0:0]
    subdistrict_id = chosen["subdistrict_id"].iloc[0] if chosen["subdistrict_id"].nunique() == 1 else ""
    # the district the label falls in: the selected place's, which can differ from the coded one
    district_id = chosen["district_id"].iloc[0] if len(chosen) else inc["district_id_2011"]
    in_district = all_places[(all_places.state_id == inc["state_id"]) & (all_places.district_id == district_id)]
    if precision in AT_PLACE:
        st.markdown(f"**Selected:** {labels_by_shrid.get(choice, '— none: pick a candidate or search —')}")
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
        named = [i for i, n in sub_names.items() if isinstance(n, str) and norm(n) in {norm(x) for x in names}]
        start = saved or subdistrict_id or (named[0] if named else "")
        subdistrict_id = st.selectbox("Subdistrict (defaults to the selected village's)", options, key=key("subdistrict"),
                                      index=options.index(start) if start in options else 0,
                                      format_func=lambda s: sub_names.get(s, "choose…"))
        if subdistrict_id:  # the label's district follows the chosen subdistrict
            district_id = sub_district[subdistrict_id]
            in_district = all_places[(all_places.state_id == inc["state_id"]) & (all_places.district_id == district_id)]
    place_type = st.selectbox("What kind of place is it?", PLACE_TYPES, key=key("type"),
                              index=PLACE_TYPES.index(default("place_type", "village")))
    reason = st.selectbox("If not a village or town: why?", REASONS, key=key("reason"),
                          index=REASONS.index(default("reason", "")))
    found_via = st.multiselect("How did you find it?", FOUND_VIA, key=key("via"),
                               default=[v for v in default("found_via", "").split("|") if v])
    cut_off = st.checkbox("Summary is cut off (labeled from the SATP page)", key=key("cut_off"),
                          value=default("summary_cut_off", "False") == "True")
    notes = st.text_input("Notes", key=key("notes"), value=default("notes", ""))

    if st.button("Save and go to next", type="primary", use_container_width=True):
        if not labeler:
            st.error("Enter your initials in the sidebar first.")
        elif precision in AT_PLACE and not choice:
            st.error("Pick a village or town, or choose a coarser unit.")
        elif precision == "subdistrict" and not subdistrict_id:
            st.error("Choose the subdistrict.")
        else:
            row = chosen.iloc[0] if precision in AT_PLACE else None
            if precision in ("district", "state or unknown"):
                subdistrict_id = ""
            sub_name = in_district.loc[in_district.subdistrict_id == subdistrict_id, "subdistrict_name"]
            save_label({
                "incident_number": iid, "split": split, "order": inc["order"], "precision": precision,
                "shrid2": "|".join(chosen["shrid2"]) if row is not None else "",  # several IDs for a whole split town
                "place_name": row["name"] if row is not None else "",
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
                "summary_cut_off": str(cut_off), "notes": notes, "labeler": labeler, "labeled_at": datetime.now().isoformat(timespec="seconds"),
            })
            st.session_state.pos = min(len(queue) - 1, st.session_state.pos + 1)
            st.rerun()
