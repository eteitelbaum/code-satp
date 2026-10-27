# Location labeling

Tools for building the location answer set (issue #7): a sample of test and validation incidents, each labeled with the most precise Census 2011 unit where the event happened. The labels replace the hand-entered coordinates as the reference for scoring location extraction.

## Running it

From the repository root:

```bash
Rscript location-labeling/build_lookup.R              # SHRUG lookup tables and district crosswalk
python location-labeling/draw_sample.py               # incidents to label
streamlit run location-labeling/app.py                # labeling app
```

`build_lookup.R` needs the SHRUG village polygons in `village-matching/data-large/` (not in the repository). Python packages are in `requirements.txt`.

| File | Contents |
|---|---|
| `build_lookup.R` | Maps each coded district to its 2011 Census district (by name, then by where the hand-entered coordinates fall); lists the SHRUG villages and towns in those districts, with simplified boundaries for the app's coordinate lookup |
| `district_overrides.csv` | Hand fixes to the district crosswalk |
| `draw_sample.py` | Reproduces the temporal split of the location models and orders the validation and test incidents that name a place below the district |
| `app.py` | The labeling app; labels save to `labels/labels.csv` |

Generated data (`data/`) and labels (`labels/`) are not committed.

## Labeling protocol

For each incident, record **where the event happened**, as precisely as the report allows: **in** a village or town, **near** a village or town, or, failing that, the subdistrict or district. The SATP coding manual covers state and district only; these rules are new for this answer set.

1. **Label where the event happened**, not where victims lived, where forces came from, or where bodies were taken.
2. **Police stations, camps, colonies, markets, neighborhoods:** **in** the village or town they sit in. For a town listed in several subdistrict parts, choose the part if you can tell, otherwise the whole town.
3. **"Near X," or a forest, hill, or river tied to X:** **near** X.
4. **Forests, hills, or areas not tied to a named place:** the subdistrict, or the district if that is all the report supports.
5. **Places SHRUG does not list:** find the spot on a map and paste its coordinates into **Locate by coordinates**. Use the map to find where a place is and the SHRUG name to choose the unit: if a nearby SHRUG village has the matching name, choose it; otherwise take the village or town that contains the point.
6. **Several plausible candidates within a few kilometers:** pick the one with the best evidence and record the alternative in the notes. Use the subdistrict only when you cannot place the event that closely.
7. **The hand-entered pin:** decide without it first, and note whether it changed your answer.
8. **Hard cases:** say what you were unsure about in the notes.

What to record:

- **Where it happened:** in or near a village or town, subdistrict, district, or state/unknown
- **Place type:** what the place in the report is (village, town, police station, camp, forest, hill, river, road, other)
- **Reason**, if coarser than a village or town: hamlet, local name, area only, camp, not found, other
- **How you found it:** shortlist, search, Google Maps, SATP page, pin
- **Summary cut off:** check this if the coded summary is truncated and you labeled from the SATP source page

## Sources

Village and town boundaries and names: SHRUG (Asher, Lunt, Matsuura, and Novosad 2021), CC BY-NC-SA 4.0. Incident reports: [South Asia Terrorism Portal](https://www.satp.org); see SATP's [terms of use](https://www.satp.org/disclaimer).
