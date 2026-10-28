"""Fetch SATP's monthly Maoist-insurgency pages and split them into entries (GitHub #8, #10).

By default fetches the months the location sample and the date checks need: each
sample incident's date and incident-number date, and both dates of every incident
in data/date_corrections.csv, within 2005-2016. Pass months to fetch others:
python location-labeling/fetch_satp.py 2016-01 2016-02

Pages are fetched one at a time with a delay and cached as raw HTML; a cached page
is not fetched again. The HTML and the entries stay local (location-labeling/data
is gitignored). See SATP's terms of use before sharing either.

Run from the repository root after draw_sample.py:  python location-labeling/fetch_satp.py

Output: location-labeling/data/satp_pages/YYYY-MM.html and location-labeling/data/satp_entries.csv
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

PAGES = Path("location-labeling/data/satp_pages")
OUT = "location-labeling/data/satp_entries.csv"
BASE = "https://www.satp.org/terrorist-activity/india-maoistinsurgency"
DELAY = 5  # seconds between requests
HEADERS = {"User-Agent": "code-satp research (https://github.com/eteitelbaum/code-satp)"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def id_month(incident_number):
    """Month encoded in an incident number (state + mmddyy + sequence), as YYYY-MM."""
    n = incident_number.zfill(10)
    return f"20{n[6:8]}-{n[2:4]}"


def page_url(month):
    year, mm = month.split("-")
    return f"{BASE}-{MONTHS[int(mm) - 1]}-{year}"


# --- months to fetch ---
if len(sys.argv) > 1:
    months = set(sys.argv[1:])
else:
    sample = pd.read_csv("location-labeling/data/sample.csv", dtype=str)
    fixes = pd.read_csv("data/date_corrections.csv", dtype=str)
    months = (
        set(sample["date"].str[:7])
        | set(sample["incident_number"].map(id_month))
        | set(fixes["date_entered"].dropna().str[:7])
        | set(fixes["incident_number"].map(id_month))
    )
    months = {m for m in months if re.fullmatch(r"20(0[5-9]|1[0-6])-(0[1-9]|1[0-2])", m)}
months = sorted(months)

# --- fetch the pages not yet cached ---
PAGES.mkdir(parents=True, exist_ok=True)
todo = [m for m in months if not (PAGES / f"{m}.html").exists()]
print(f"{len(months)} months; {len(todo)} to fetch")
for i, month in enumerate(todo):
    if i:
        time.sleep(DELAY)
    response = requests.get(page_url(month), headers=HEADERS, timeout=60)
    if response.status_code != 200:
        print(f"{month}: HTTP {response.status_code}, skipped")
        continue
    (PAGES / f"{month}.html").write_text(response.text, encoding="utf-8")
    print(f"{month}: fetched")

# --- split each cached page into entries: a date cell beside the full text ---
rows = []
for path in sorted(PAGES.glob("*.html")):
    month = path.stem
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for div in soup.find_all("div", class_="more"):
        date_cell = div.find_parent("tr").find("td", style="width: 15%;")
        day = re.search(r"(\d+)\s*$", date_cell.get_text(" ", strip=True).replace("\xa0", " "))
        text = div.get_text(" ", strip=True).removesuffix("Read less...").strip()
        rows.append({
            "month": month,
            "entry_id": div["id"],
            "date": f"{month}-{int(day.group(1)):02d}" if day else None,
            "text": re.sub(r"\s+", " ", text),
            "url": page_url(month),
        })

entries = pd.DataFrame(rows)
entries.to_csv(OUT, index=False)
print(f"{len(entries)} entries from {entries['month'].nunique()} pages; {entries['date'].isna().sum()} without a date")
