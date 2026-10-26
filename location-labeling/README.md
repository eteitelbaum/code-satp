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

For each incident, record **where the event happened**, as precisely as the report allows: **in** a village or town, **near** a village or town, or, failing that, the subdistrict or district. Record what the report supports and decide how to score it later: a "near" label can be scored at the village's location (distance) or coarsened to its subdistrict (exact unit). The SATP coding manual covers state and district only; these rules are new for this answer set. Rules marked *(proposed)* are not yet settled.

1. **Where the event happened.** When a report names several places, label the place where the attack, clash, arrest, or surrender took place, not where victims lived, where forces came from, or where bodies were taken.
2. **Police stations, camps, colonies, schools, markets, city neighborhoods:** **in** the village or town they sit in, if you can identify it. SHRUG splits some cities into pieces by subdistrict (Greater Hyderabad appears as several "GHMC" towns); choose the piece the neighborhood is in.
3. **"Near X," or a forest, hill, or river tied to X** ("Karulai forest range," "the hills near X") *(proposed):* **near** X.
4. **Forests, hills, rivers not tied to a named village:** the subdistrict.
5. **Places SHRUG does not list** (hamlets, neighborhoods, local names): find the spot on Google Maps, copy its coordinates (right-click, then click the coordinates), and paste them into **Locate by coordinates**; the app selects the SHRUG village or town that contains the point. Label **in** that place if the spot is inside it, **near** it otherwise; if you cannot find the spot, use the subdistrict with the reason `hamlet` or `local name`.
6. **The hand-entered pin:** decide without it first. Show it only if you are stuck, and note whether it changed your answer.
7. **Hard cases:** say what you were unsure about in the notes.

What to record:

- **Where it happened:** in a village or town, near a village or town, subdistrict, district, or state/unknown. Choosing "subdistrict" fills in the subdistrict of the village you selected; change it if needed.
- **Place type:** what the place in the report is (village, town, police station, camp, forest, hill, river, road, other). This can differ from SHRUG's type: SHRUG lists some 2011 towns, such as Nilambur, as villages.
- **Reason** (if coarser than a village or town): hamlet, local name, area only, camp, not found, other
- **How you found it:** shortlist, search, Google Maps, SATP page, pin. Pasted coordinates are saved with the label.
- **Summary cut off:** check this when the coded summary is truncated and you labeled from the SATP source page. The app warns when a summary has no final punctuation.

Candidate buttons are ranked by how closely a SHRUG name's spelling matches a coded name (1.00 = same spelling). The score is not a probability; a close spelling can be an unrelated village.

## Sources

Village and town boundaries and names: SHRUG (Asher, Lunt, Matsuura, and Novosad 2021), CC BY-NC-SA 4.0. Incident reports: [South Asia Terrorism Portal](https://www.satp.org); see SATP's [terms of use](https://www.satp.org/disclaimer).
