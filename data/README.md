# Data

## How the data were made

The dataset covers India's Maoist insurgency from 2005 to 2016. A team of trained coders read incident reports catalogued by the [South Asia Terrorism Portal](https://www.satp.org) (SATP) and entered each incident through a Google Form, following a project codebook that was refined between 2011 and 2016 as new kinds of cases came up. New coders trained on the codebook and coded overlapping assignments with an experienced evaluator before working on their own. Ambiguous cases were logged and resolved in weekly meetings. A large share of the data was double-coded, with discrepancies (under 5% of observations) resolved by a senior coder. A final review harmonized coding rules across years.

## Coding rules in brief

- **Unit:** one violent incident. Reports of policy changes, announcements by Maoists or security forces, strike calls, and protests are not coded.
- **Multiple incidents in one report** are coded separately when they happened in different places or were different operations (e.g., a Maoist attack and a separate police arrest on the same day). Several actions within one attack are one incident with several action and target types.
- **Perpetrator:** Maoist, Security, Unknown, or Non-Maoist Armed Group. Unknown when the report does not say who initiated the violence (e.g., "Maoists and security forces exchanged fire").
- **Action types** (up to three, in the order reported): armed assault, hijacking/kidnapping, bombing, facility/infrastructure attack, surrender, arrest, seizure.
- **Target types** (up to two): security, Maoists, civilians, government officials, government infrastructure, private property, mining company, NGO, no target.
- **Counts:** fatalities and injuries by group (civilian, security, Maoist, government official), arrests and surrenders by rank, abduction victims, and property damage.
- **Incident number:** two-digit state code + date (mmddyy) + two-digit counter for that state and day.
- **Incident summary:** one or two sentences, mostly pasted from SATP.

For the full rules, see the coding manual and the description of the coding procedure in [`documentation/`](../documentation).

## Details not in the original documentation

- **Source pages.** Incidents were coded from SATP's pages as they stood at the time, including state timelines from an earlier version of the site. Some coded incidents no longer appear on the current site, and summaries were trimmed (lead-ins, report dates, and source credits removed), so they do not match SATP's current text word for word.
- **Location variables.** Besides state and district, coders recorded the block, village, other named places (police stations, forests, camps), and constituency. Latitude and longitude were added later by a separate team, who looked up each village name in Google Maps. Treat the coordinates with caution: some points are fallbacks reused for many different villages rather than village locations.
- **Incident numbers** are stored as numbers in the CSV files, so the leading zero of state codes 01–09 is dropped (e.g., `0101070701` appears as `101070701`).
- **Dates** are accurate to about a day. Where a coded date differs from SATP's, the coders often dated a different stage of the same event (the abduction rather than the killing, the encounter rather than a later death), and SATP sometimes files an entry under the wrong day, month, or year. Corrections are tracked in issue #10.
- **Duplicate removal.** `wrangle_satp.qmd` drops every row whose summary text appears more than once (808 of 10,729 rows), leaving 9,921 incidents. Most of these are reports the coders split into several incidents, or incidents coded twice. They are kept in `satp_duplicates.csv` for recoding.

## Files

| File | Contents |
|---|---|
| `satp-dataset.xlsx` | The coded data as entered (sheet `analysis`) |
| `wrangle_satp.qmd` | Cleans the coded data; writes `satp_clean.csv` |
| `satp_clean.csv` | Cleaned incident-level data, one row per incident |
| `satp_duplicates.csv` | Rows dropped as duplicates, kept for recoding |
| `check_dates.R` | Lists incidents whose date disagrees with the date in the incident number; writes `date_corrections.csv` |
| `satp_dates.csv` | Dates settled from SATP's text (written by `location-labeling/satp_dates.py`) |
| `summary_corrections.csv` | Summaries replaced because they held text from another event |
| `select_vars_for_models.qmd` | Writes one CSV per modeling task from `satp_clean.csv` |
| `perpetrator.csv`, `action_type.csv`, `target_type.csv` | Classification tasks |
| `deaths.csv`, `injuries.csv`, `arrests.csv`, `surrenders.csv`, `abductions.csv`, `property_damage.csv` | Count and extraction tasks |
| `location_info.csv`, `identifying_info.csv` | Location fields; incident number and date |
| `location_info_augmented.csv` | Location fields with block and other areas combined, used by the current location models |
| `satp_location_matched.csv`, `satp_location_matched_lat_long.csv` | Earlier location training data |
| `satp_classification.csv` | Earlier export of the cleaned data |
| `gemma_counts_output.csv` | Count extraction output from Gemma |
| `shrid_loc_names.csv` | SHRUG Census 2011 place names (CC BY-NC-SA 4.0) |

Incident reports come from SATP; see SATP's [terms of use](https://www.satp.org/disclaimer).
