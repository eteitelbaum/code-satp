# Check incident dates against the date encoded in each incident number (issue #10).
#
# Incident numbers encode state, date and sequence: 2106260901 is state 21 (West Bengal),
# 06-26-09, entry 01. Compares that date with the Date and Year fields of the raw workbook, for the
# incidents in satp_clean.csv, and writes data/date_corrections.csv: one row per incident whose
# date needs attention, with the date to use where the evidence is clear. wrangle_satp.qmd applies
# those dates. The rest stay as entered, marked unverified, until the incidents are linked to
# their SATP entries (#3). Reads the raw workbook, not satp_clean.csv, so rerunning it after the
# fix gives the same list.
#
# Run from the repository root: Rscript data/check_dates.R

library(tidyverse)

kept <- read_csv("data/satp_clean.csv", col_types = cols(.default = "c"))$incident_number

satp <- readxl::read_excel("data/satp-dataset.xlsx", sheet = "analysis", col_types = "text") |>
  select(incident_number = `Incidence Number`, year = Year, date = Date) |>
  filter(!is.na(incident_number), incident_number %in% kept)

checked <- satp |>
  mutate(
    number = format(as.numeric(incident_number), scientific = FALSE, trim = TRUE),
    state_code = str_sub(number, 1, -9),
    id_date = make_date(
      2000 + as.integer(str_sub(number, -4, -3)),  # yy
      as.integer(str_sub(number, -8, -7)),         # mm
      as.integer(str_sub(number, -6, -5))          # dd
    ),
    # Excel serial dates; one date was typed as text ("11/21/2016") and isn't read
    entered = as_date(as.numeric(date), origin = "1899-12-30"),
    year = as.integer(year),
    id_year_agrees = coalesce(year(id_date) == year, FALSE),
    problem = case_when(
      entered == id_date ~ NA_character_,
      is.na(entered) ~ "date missing or unreadable",
      year(entered) == year & !id_year_agrees ~ "incident number off",
      month(entered) == month(id_date) & day(entered) == day(id_date) ~ "wrong year in date",
      year(entered) == year(id_date) & month(entered) == day(id_date) & day(entered) == month(id_date) ~
        "day and month swapped in date",
      TRUE ~ "date and incident number differ"
    ),
    # The date to use: from the incident number when it and the year column agree on the year; the
    # entered date when it and the year column agree instead; blank (unverified) otherwise
    corrected = case_when(
      problem %in% c("date missing or unreadable", "wrong year in date", "day and month swapped in date") & id_year_agrees ~ id_date,
      problem == "incident number off" ~ entered
    ),
    source = case_when(
      problem == "incident number off" ~ "date and year column",
      !is.na(corrected) ~ "incident number and year column",
      TRUE ~ "unverified: check against SATP"
    ),
    days_apart = as.integer(entered - id_date)
  )

corrections <- checked |>
  filter(!is.na(problem)) |>
  transmute(incident_number = number, state_code, year, date_entered = entered, id_date,
            days_apart, id_year_agrees, problem, date_to_use = corrected, source) |>
  arrange(problem, incident_number)

write_csv(corrections, "data/date_corrections.csv", na = "")

cat("Incidents:", nrow(checked), "\n")
cat("Date matches incident number:", sum(is.na(checked$problem)), "\n")
corrections |>
  count(problem, source) |>
  print()
