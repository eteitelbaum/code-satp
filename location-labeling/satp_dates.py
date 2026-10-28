"""Settle incident dates from SATP's text (GitHub #10).

Two groups of incidents get SATP's date:
- the date discrepancies in data/date_corrections.csv (the date column and the
  incident number disagree), when their SATP entry was found;
- location-sample incidents dated a day off from SATP, when SATP's text and its
  listing agree with each other.

SATP's date is the event date its text gives ("on March 11", "On 15 November",
"night of June 30"), skipping report dates ("reports The Hindu on May 2"), or its
listing date when the text gives none. A date in satp_link_review.csv overrides
both (entries SATP filed under the wrong year). Incident numbers are not changed.

Run from the repository root after link_satp.py:  python location-labeling/satp_dates.py

Output: data/satp_dates.csv, read by data/wrangle_satp.qmd
"""

import re

import pandas as pd

OUT = "data/satp_dates.csv"
FOUND = ["matched", "matched: a day apart", "review: other day", "reviewed: confirmed", "reviewed: relinked"]

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
M = "|".join(MONTHS)
EVENT = re.compile(rf"\b(?:on|of)\s+(?:({M})\s+(\d{{1,2}})|(\d{{1,2}})\s+({M}))\b", re.IGNORECASE)
REPORT = re.compile(r"\b(?:reports?|reported|said|told|informed)\b[^.]{0,60}$", re.IGNORECASE)


def text_date(text, year):
    """First event date the text gives, in the year of the page it is filed on."""
    for m in EVENT.finditer(text):
        if REPORT.search(text[:m.start()][-80:]):
            continue
        month, day = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        month = [x.lower() for x in MONTHS].index(month.lower()) + 1
        return pd.to_datetime(f"{year}-{month:02d}-{int(day):02d}", errors="coerce")
    return None


links = pd.read_csv("location-labeling/data/satp_links.csv", dtype=str)
entries = pd.read_csv("location-labeling/data/satp_entries.csv", dtype=str)
review = pd.read_csv("location-labeling/satp_link_review.csv", dtype=str)
fixes = pd.read_csv("data/date_corrections.csv", dtype=str)

links = links[links["status"].isin(FOUND)].merge(
    entries[["month", "entry_id", "text"]], left_on=["entry_month", "entry_id"], right_on=["month", "entry_id"]
)
links["text_date"] = [text_date(t, m[:4]) for t, m in zip(links["text"], links["entry_month"])]
links["listing_date"] = pd.to_datetime(links["entry_date"])
links["satp_date"] = links["text_date"].fillna(links["listing_date"])
links["basis"] = links["text_date"].notna().map({True: "SATP text", False: "SATP listing"})

hand = review.dropna(subset=["date"]).set_index("incident_number")["date"]
by_hand = links["incident_number"].isin(hand.index)
links.loc[by_hand, "satp_date"] = pd.to_datetime(links.loc[by_hand, "incident_number"].map(hand))
links.loc[by_hand, "basis"] = "hand review"

# --- which incidents take SATP's date ---
discrepancy = links["incident_number"].isin(fixes["incident_number"])
coded = pd.to_datetime(links["date"])
a_day_off = (
    (links["status"] == "matched: a day apart")
    & (links["text_date"] == links["listing_date"])
    & (links["satp_date"] != coded)
)
links["reason"] = None
links.loc[a_day_off, "reason"] = "a day off in the location sample"
links.loc[discrepancy, "reason"] = "date and incident number disagree"

out = links[links["reason"].notna()].copy()
out["entry"] = out["entry_month"] + " " + out["entry_id"]
out = out.rename(columns={"date": "date_entered"})
cols = ["incident_number", "date_entered", "id_date", "entry", "listing_date", "text_date", "satp_date", "basis", "reason"]
for c in ["listing_date", "text_date", "satp_date"]:
    out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
out[cols].sort_values("incident_number").to_csv(OUT, index=False)

print(out.groupby(["reason", "basis"]).size().to_string())
d = out[out["reason"] == "date and incident number disagree"]
print(f"discrepancies settled: {len(d)} of {len(fixes)}; SATP date equals the incident-number date: "
      f"{(d['satp_date'] == d['id_date']).sum()}, the date column: {(d['satp_date'] == d['date_entered']).sum()}, neither: "
      f"{((d['satp_date'] != d['id_date']) & (d['satp_date'] != d['date_entered'])).sum()}")
