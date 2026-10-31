"""Match UCDP GED events to sample incidents (GitHub #18).

GED (Sundberg and Melander 2013; version 26.1) codes events with at least one death, with
coordinates and a precision code (`where_prec`). Where a sample incident matches a GED event,
GED's point is a baseline for the location methods.

Kept from GED: India, the dyads of the Maoist conflict (government vs CPI-Maoist, CPI-Maoist vs
civilians, PLFI vs civilians), starting from DAYS before the first sample incident through 2016.

A GED event is a candidate for an incident when the incident's date (SATP's, from link_satp.py,
else the date in its incident number) falls within the event's dates, give or take DAYS, and
GED's point lies in the incident's 2011 state. Candidates are ranked by
1. district: GED's point lies in the incident's 2011 district, or GED's district name is the
   coded one (only the name for points at state precision or coarser);
2. deaths: GED's best estimate against the coded total;
3. days outside GED's dates.
An incident coded with no deaths is a candidate only in the same district.
Pairs are then taken one to one, best first. A pair is "matched" when the district and deaths
agree and neither side has another candidate as good; every other pair is "check", for review by
hand in ged_match_review.csv (a decision there overrides the matching).

Download GED's CSV from https://ucdp.uu.se/downloads/ (ged261-csv.zip) and unzip it into
location-labeling/data/ged/.

Run from the repository root after link_satp.py:  python location-labeling/match_ged.py

Output: location-labeling/data/ged/matches.csv, one row per GED event with its incident, if any
"""

import re
from pathlib import Path

import pandas as pd
import shapely

ROOT = Path(__file__).parent
DATA = ROOT / "data"
GED = DATA / "ged" / "GEDEvent_v26_1.csv"
REVIEW = ROOT / "ged_match_review.csv"
OUT = DATA / "ged" / "matches.csv"
DYADS = ["Government of India - CPI-Maoist", "CPI-Maoist - Civilians", "PLFI - Civilians"]
END = "2016-12-31"
DAYS = 2


def norm(name):
    return re.sub(r"[^a-z]", "", re.sub(r"\bdistrict\b", "", str(name).lower()))


def main():
    sample = pd.read_csv(DATA / "sample.csv", dtype=str)
    links = pd.read_csv(DATA / "satp_links.csv", dtype=str)[["incident_number", "entry_date", "id_date", "status"]]
    deaths = pd.read_csv("data/satp_clean.csv", dtype=str, usecols=["incident_number", "total_fatalities"])
    inc = sample.merge(links, on="incident_number", how="left").merge(deaths, on="incident_number", how="left")
    found = ~inc.status.fillna("").str.contains("not found")
    inc["satp_date"] = pd.to_datetime(inc.entry_date.where(found & inc.entry_date.notna(), inc.id_date))
    inc["deaths"] = pd.to_numeric(inc.total_fatalities, errors="coerce").fillna(0).astype(int)

    ged = pd.read_csv(GED, low_memory=False)
    ged = ged[(ged.country == "India") & ged.dyad_name.isin(DYADS)].copy()
    ged["start"], ged["end"] = pd.to_datetime(ged.date_start), pd.to_datetime(ged.date_end)
    start = pd.to_datetime(inc.date).min() - pd.Timedelta(days=DAYS)
    ged = ged[(ged.start >= start) & (ged.start <= END)].reset_index(drop=True)
    ged["cites_satp"] = ged.source_article.fillna("").str.contains(r"\bSATP\b|South Asia Terrorism Portal", case=False)

    # GED's point in the 2011 Census units
    subs = pd.read_parquet(DATA / "subdistrict_shapes.parquet")
    tree = shapely.STRtree(shapely.from_wkb(subs["wkb"].values))
    pi, si = tree.query(shapely.points(ged.longitude.values, ged.latitude.values), predicate="intersects")
    hit = subs.iloc[si][["state_id", "district_id", "subdistrict_id"]].astype(str).set_index(pd.Index(pi))
    hit = hit[~hit.index.duplicated()].reindex(range(len(ged)))
    ged["ged_state_id"], ged["ged_district_id"], ged["ged_subdistrict_id"] = hit.state_id.values, hit.district_id.values, hit.subdistrict_id.values

    # candidates
    pairs = ged.reset_index().rename(columns={"index": "g"}).merge(
        inc.reset_index().rename(columns={"index": "s"}), left_on="ged_state_id", right_on="state_id")
    lo, hi = pairs.start - pd.Timedelta(days=DAYS), pairs.end + pd.Timedelta(days=DAYS)
    pairs = pairs[(pairs.satp_date >= lo) & (pairs.satp_date <= hi)].copy()
    pairs["days_off"] = ((pairs.start - pairs.satp_date).dt.days.clip(lower=0)
                         + (pairs.satp_date - pairs.end).dt.days.clip(lower=0))
    name = (pairs.adm_2.map(norm) == pairs.district.map(norm)) | (pairs.adm_2.map(norm) == pairs.district_name_2011.map(norm))
    point = (pairs.where_prec <= 3) & (pairs.ged_district_id == pairs.district_id_2011)
    pairs["same_district"] = name | point
    pairs["deaths_off"] = (pairs.best - pairs.deaths).abs()
    pairs = pairs[pairs.same_district | (pairs.deaths > 0)]
    pairs = pairs.sort_values(["same_district", "deaths_off", "days_off"], ascending=[False, True, True])
    pairs["rank"] = list(zip(~pairs.same_district, pairs.deaths_off, pairs.days_off))

    # one to one, best first; ties on either side are flagged
    used_g, used_s, rows = set(), set(), []
    for p in pairs.itertuples():
        if p.g in used_g or p.s in used_s:
            continue
        rivals = pairs[((pairs.g == p.g) | (pairs.s == p.s)) & (pairs["rank"] == p.rank)
                       & ~pairs.g.isin(used_g) & ~pairs.s.isin(used_s)]
        clean = p.same_district and p.deaths_off == 0 and len(rivals) == 1
        rows.append({"g": p.g, "incident_number": p.incident_number, "status": "matched" if clean else "check",
                     "same_district": p.same_district, "deaths_off": p.deaths_off, "days_off": p.days_off,
                     "rivals": len(rivals) - 1})
        used_g.add(p.g), used_s.add(p.s)
    match = pd.DataFrame(rows, columns=["g", "incident_number", "status", "same_district", "deaths_off", "days_off", "rivals"])

    out = ged.reset_index().rename(columns={"index": "g"}).merge(match, on="g", how="left")
    out["status"] = out.status.fillna("unmatched")
    if REVIEW.exists():
        review = pd.read_csv(REVIEW, dtype=str)
        for r in review.itertuples():
            k = out.id.astype(str) == r.ged_id
            out.loc[k, "incident_number"] = r.incident_number if r.incident_number != "none" else pd.NA
            out.loc[k, "status"] = "reviewed: " + ("none" if r.incident_number == "none" else "matched")
    out = out.merge(inc[["incident_number", "split", "order", "district", "deaths", "satp_date", "incident_summary"]],
                    on="incident_number", how="left")
    cols = ["id", "status", "incident_number", "split", "order", "dyad_name", "date_start", "date_end", "satp_date",
            "adm_1", "adm_2", "district", "where_prec", "where_coordinates", "where_description", "latitude", "longitude",
            "ged_state_id", "ged_district_id", "ged_subdistrict_id", "best", "deaths", "same_district", "deaths_off",
            "days_off", "rivals", "cites_satp", "number_of_sources", "source_article", "incident_summary"]
    out[cols].sort_values(["date_start", "id"]).to_csv(OUT, index=False)

    print(f"GED events: {len(out)} (cite SATP: {out.cites_satp.sum()})")
    print(out.status.value_counts().to_string())
    m = out[out.incident_number.notna()]
    print(m.groupby("split").size().rename("matched incidents").to_string())
    lethal = inc[inc.deaths > 0]
    print(f"lethal sample incidents: {lethal.groupby('split').size().to_dict()}; "
          f"with a GED match: {lethal[lethal.incident_number.isin(m.incident_number)].groupby('split').size().to_dict()}")
    print(m.where_prec.value_counts().sort_index().rename("where_prec of matched").to_string())


if __name__ == "__main__":
    main()
