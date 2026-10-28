# Location labeling

Tools for building the location answer set (issue #7): a sample of test and validation incidents, each labeled with the most precise Census 2011 unit where the event happened. The labels replace the hand-entered coordinates as the reference for scoring location extraction.

## Running it

From the repository root:

```bash
Rscript location-labeling/build_lookup.R              # SHRUG lookup tables and district crosswalk
python location-labeling/draw_sample.py               # incidents to label
python location-labeling/fetch_satp.py                # SATP monthly pages (cached locally)
python location-labeling/link_satp.py                 # link each incident to its SATP entry
python location-labeling/satp_dates.py                # dates settled from SATP (#10)
python location-labeling/build_osm.py                 # OpenStreetMap's named places (download India from Geofabrik first; see the script)
python location-labeling/suggest.py validation        # model suggestions for the app (#9; needs OPENAI_API_KEY in .env)
streamlit run location-labeling/app.py                # labeling app
```

`build_lookup.R` needs the SHRUG village polygons in `village-matching/data-large/` (not in the repository). Python packages are in `requirements.txt`.

| File | Contents |
|---|---|
| `build_lookup.R` | Maps each coded district to its 2011 Census district (by name, then by where the hand-entered coordinates fall); lists the SHRUG villages and towns in those districts, with simplified boundaries for the app's coordinate lookup |
| `district_overrides.csv` | Hand fixes to the district crosswalk |
| `draw_sample.py` | Reproduces the temporal split of the location models and orders the validation and test incidents that name a place below the district |
| `fetch_satp.py` | Fetches SATP's monthly Maoist-insurgency pages for the sample's months and the date checks (#10), one at a time, and splits them into entries |
| `link_satp.py` | Links each sample incident, and each date discrepancy in `data/date_corrections.csv`, to its SATP entry by word overlap with the coded summary, within a few days of its date |
| `satp_link_review.csv` | Hand decisions on links the matching could not settle: the entry, any further entries the summary draws on, or "not found"; and the date, where SATP's is wrong |
| `satp_dates.py` | Takes SATP's date for the date discrepancies and for sample incidents a day off from SATP; writes `data/satp_dates.csv` |
| `build_osm.py` | OpenStreetMap's named settlements, forests, protected areas, and hills in the study states, under every Latin-script name (plus short forms such as "Karlapat" for "Karlapat WLS"), each tied to the SHRUG village and subdistrict it falls in; writes `data/osm_places.parquet`, and the outlines of forests and protected areas to `data/osm_areas.parquet` |
| `suggest.py` | Suggestions for the app: GPT-4o-mini lists the places in the text and their roles; each is matched by spelling to SHRUG villages, towns, and subdistricts (and to OpenStreetMap's places when SHRUG has nothing close); forests and hills that OpenStreetMap outlines also get points inside them (protocol rule 4); the top five options are ranked by spelling and context, and the model picks one with a reason. Model calls are cached in `data/suggest/`; each run's outputs are kept in `data/suggest/runs/`, and each label records the run it saw |
| `app.py` | The labeling app; labels save to `labels/labels.csv` |

Generated data (`data/`, including SATP's pages and text) and labels (`labels/`) are not committed.

## Labeling protocol

For each incident, record **where the event happened**, as precisely as the report allows: **in** a village or town, **near** a village or town, a point **in a forest or area** (rule 4), or, failing that, the subdistrict or district. Choose the most specific place the report supports before falling back to the subdistrict. The SATP coding manual covers state and district only; these rules are new for this answer set.

1. **Label where the event happened**, not where victims lived, where forces came from, or where bodies were taken.
2. **Events at police stations, camps, colonies, markets, neighborhoods:** **in** the village or town they sit in. For a town listed in several subdistrict parts, choose the part if you can tell, otherwise the whole town. "Under X Police Station (limits)" names the station's jurisdiction, not the place: use it to choose the subdistrict only when the report names nothing more specific (for forests, see rule 4).
3. **"Near X":** **near** X.
4. **Forests, hills, sanctuaries:** take the first that applies.
   1. A village or town the report names in or near the forest, if you can find it: **in** or **near** it.
   2. The forest is mapped and the report names a jurisdiction (police station, mandal, block): a point inside the part of the forest within that jurisdiction's subdistrict (**in a forest or area**).
   3. The forest is named after a place you can find ("Karlapat forest"): **near** that place.
   4. The forest is mapped: a point in the middle of it (**in a forest or area**).
   5. Otherwise: the subdistrict, or the district if that is all the report supports.

   "Mapped" means outlined on OpenStreetMap or Google Maps. Tree cover counts; legal status does not. A point in a forest usually falls outside every village, so its unit is the subdistrict (rule 6). When OpenStreetMap names the forest, the suggestions include these points and the map shows its outline; otherwise paste a point under **Locate by coordinates**.
5. **Places SHRUG does not list:** find the spot on a map and paste its coordinates into **Locate by coordinates**, and choose the SHRUG place the report names, if any, as the place the event was in or near.
6. **Point and unit:** each label saves a point for mapping (your pasted point, or else the chosen place's location) and a unit for analysis, taken from the point: the village or town it falls in, or the subdistrict when it falls outside every village (often reserved forest). The app shows both before you save.
7. **Several plausible candidates within a few kilometers:** pick the one with the best evidence and record the alternative in the notes. Use the subdistrict only when you cannot place the event that closely.
8. **The hand-entered pin:** decide without it first, and note whether it changed your answer.
9. **Hard cases:** say what you were unsure about in the notes.
10. **Order:** the app serves incidents in the sample's random order, so labels so far are a random subset. You can go back to earlier incidents, but not ahead.
11. **Skipping:** only when an incident cannot be labeled (not a single incident, or the text does not describe the coded event), with the reason recorded. Skipped incidents count in the denominator.

What to record:

- **Where it happened:** in or near a village or town, in a forest or area (a point), subdistrict, district, or state/unknown
- **Place type:** what the place in the report is (village, town, police station, camp, forest, hill, river, road, other)
- **Reason**, if coarser than a village or town: hamlet, local name, area only, camp, not found, other
- **How you found it:** suggestion, shortlist, search, Google Maps, OpenStreetMap, SATP page, pin
- **Summary cut off:** check this if the coded summary is truncated and you labeled from the SATP source page

## Sources

Village and town boundaries and names: SHRUG (Asher, Lunt, Matsuura, and Novosad 2021), CC BY-NC-SA 4.0. Hamlets, localities, forests, and hills: © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, ODbL, via [Geofabrik](https://download.geofabrik.de/asia/india.html). Incident reports: [South Asia Terrorism Portal](https://www.satp.org); see SATP's [terms of use](https://www.satp.org/disclaimer).
