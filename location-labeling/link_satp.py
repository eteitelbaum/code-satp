"""Link each sample incident, and each date discrepancy, to its SATP entry (GitHub #8, #10).

The incidents are the location sample and the incidents in data/date_corrections.csv
(split "date check" when they are not in the sample).

Candidates are the entries dated within WINDOW days of the incident's date or of
the date encoded in its incident number (the two disagree for misdated incidents,
#10). Each candidate is scored by word overlap: the share of the coded summary's
content words that appear in the entry. The best candidate is the link. When no
candidate reaches MATCH, the search widens to WIDE days, which finds entries dated
a month off (an error the incident number shares, so the #10 check misses it).
A link needs review when its overlap is below MATCH, when the runner-up is within
MARGIN of it, or when it is dated more than a day from both of the incident's dates.

Reviewed links are recorded in location-labeling/satp_link_review.csv (decision,
entry, any further entries the coded summary draws on, and a note) and override the
automatic link.

Run from the repository root after fetch_satp.py:  python location-labeling/link_satp.py

Output: location-labeling/data/satp_links.csv
"""

import re

import pandas as pd

OUT = "location-labeling/data/satp_links.csv"
WINDOW = 3  # days either side of each date
WIDE = 45  # second pass, for incidents with no match within WINDOW
MATCH = 0.6
MARGIN = 0.1

STOP = set("""
a an and are as at be been but by for from had has have he her his in into is it its
of on or that the their them they this to was were which who with will would after
also over under near said district police reports area village station limits
""".split())


def words(text):
    return {w for w in re.findall(r"[a-z]+", str(text).lower()) if len(w) > 2 and w not in STOP}


def id_date(incident_number):
    """Date encoded in an incident number (state + mmddyy + sequence)."""
    n = incident_number.zfill(10)
    return pd.to_datetime(f"20{n[6:8]}-{n[2:4]}-{n[4:6]}", errors="coerce")


sample = pd.read_csv("location-labeling/data/sample.csv", dtype=str)
fixes = pd.read_csv("data/date_corrections.csv", dtype=str)
fixes = fixes[~fixes["incident_number"].isin(sample["incident_number"])]
summaries = pd.read_csv("data/satp_clean.csv", dtype=str, usecols=["incident_number", "incident_summary"])
checks = fixes.merge(summaries, on="incident_number").rename(columns={"date_entered": "date"})
checks["split"] = "date check"
cols = ["split", "order", "incident_number", "date", "incident_summary"]
incidents = pd.concat([sample, checks], ignore_index=True)[cols]
incidents["date"] = pd.to_datetime(incidents["date"])
incidents["id_date"] = incidents["incident_number"].map(id_date)

entries = pd.read_csv("location-labeling/data/satp_entries.csv", dtype=str)
entries["date"] = pd.to_datetime(entries["date"])
entries["words"] = entries["text"].map(words)

# --- score the candidates for each incident ---
def score(inc, summary, days):
    window = pd.Timedelta(days=days)
    near = pd.Series(False, index=entries.index)
    for d in (inc.date, inc.id_date):
        if pd.notna(d):
            near |= (entries["date"] - d).abs() <= window
    cands = entries[near]
    scores = sorted(
        ((len(summary & e.words) / len(summary), e.Index) for e in cands.itertuples()),
        reverse=True,
    ) if summary else []
    return cands, scores


rows = []
for inc in incidents.itertuples():
    summary = words(inc.incident_summary)
    cands, scores = score(inc, summary, WINDOW)
    if not scores or scores[0][0] < MATCH:
        cands, scores = score(inc, summary, WIDE)
    row = {
        "split": inc.split, "order": inc.order, "incident_number": inc.incident_number,
        "date": inc.date.date() if pd.notna(inc.date) else None, "id_date": inc.id_date.date() if pd.notna(inc.id_date) else None,
        "n_candidates": len(cands),
    }
    if scores:
        best, i = scores[0]
        e = entries.loc[i]
        row.update({
            "entry_month": e["month"], "entry_id": e["entry_id"], "entry_date": e["date"].date(),
            "overlap": round(best, 3), "runner_up": round(scores[1][0], 3) if len(scores) > 1 else None,
            "url": e["url"],
        })
    rows.append(row)

links = pd.DataFrame(rows)

# --- flag links to review ---
def days_apart(a, b):
    return (pd.to_datetime(links[a]) - pd.to_datetime(links[b])).dt.days.abs()

gap = pd.concat([days_apart("entry_date", "date"), days_apart("entry_date", "id_date")], axis=1).min(axis=1)
links["status"] = "matched"
links.loc[gap == 1, "status"] = "matched: a day apart"
links.loc[gap > 1, "status"] = "review: other day"
links.loc[links["runner_up"].fillna(0) > links["overlap"] - MARGIN, "status"] = "review: close runner-up"
links.loc[links["overlap"] < MATCH, "status"] = "review: low overlap"
links.loc[links["n_candidates"] == 0, "status"] = "no candidates"

# --- apply the hand review ---
review = pd.read_csv("location-labeling/satp_link_review.csv", dtype=str).set_index("incident_number")
review = review[review.index.isin(incidents["incident_number"])]
entry_cols = ["entry_month", "entry_id", "entry_date", "overlap", "runner_up", "url"]
links = links.set_index("incident_number")
links[["also", "note"]] = review[["also", "note"]]
for n, r in review.iterrows():
    links.loc[n, entry_cols] = None
    links.loc[n, "status"] = f"reviewed: {r['decision']}"
    if pd.notna(r["entry_id"]):
        e = entries[(entries["month"] == r["entry_month"]) & (entries["entry_id"] == r["entry_id"])].iloc[0]
        summary = words(incidents.loc[incidents["incident_number"] == n, "incident_summary"].iloc[0])
        links.loc[n, ["entry_month", "entry_id", "entry_date", "url"]] = [e["month"], e["entry_id"], e["date"].date(), e["url"]]
        links.loc[n, "overlap"] = round(len(summary & e["words"]) / len(summary), 3)
links = links.reset_index()
links["shared_entry"] = links.groupby(["entry_month", "entry_id"])["incident_number"].transform("size").gt(1)

links.to_csv(OUT, index=False)
print(links.groupby(["split", "status"]).size().to_string())
print(f"shared entries: {links['shared_entry'].sum()}")
