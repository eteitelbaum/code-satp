# Build the SHRUG lookup tables for the location labeling app
#
# Run from the repository root:  Rscript location-labeling/build_lookup.R
#
# Inputs
#   village-matching/data-large/shrug-pc11-village-poly-gpkg/village.gpkg  SHRUG 2011 village/town polygons
#   village-matching/data-large/shrug-pc11subdist-poly-gpkg/subdistrict.gpkg  SHRUG 2011 subdistrict polygons
#   data/shrid_loc_names.csv                                               SHRUG place, subdistrict, district names
#   data/satp_clean.csv                                                    coded incidents (state, district, pins)
#   location-labeling/district_overrides.csv                               hand fixes to the crosswalk (optional)
#
# Outputs (location-labeling/data/)
#   district_crosswalk.csv  coded state/district -> 2011 district, with how it was matched
#   places.parquet          SHRUG villages and towns in the matched 2011 districts, with a point for the map
#   pin_places.parquet      the SHRUG place each incident's hand-entered pin falls in
#   place_shapes.parquet    simplified village/town boundaries, for the app's coordinate lookup
#   subdistrict_shapes.parquet  simplified subdistrict boundaries in the study states, for points outside every village
#
# SHRUG data are CC BY-NC-SA 4.0 (Asher, Lunt, Matsuura and Novosad 2021).

library(sf)
library(dplyr)
library(readr)
library(stringr)
library(stringdist)
library(arrow)

sf_use_s2(FALSE)

gpkg_file     <- "village-matching/data-large/shrug-pc11-village-poly-gpkg/village.gpkg"
subdist_file  <- "village-matching/data-large/shrug-pc11subdist-poly-gpkg/subdistrict.gpkg"
names_file    <- "data/shrid_loc_names.csv"
incident_file <- "data/satp_clean.csv"
override_file <- "location-labeling/district_overrides.csv"
out_dir       <- "location-labeling/data"
dir.create(out_dir, showWarnings = FALSE)

# lower case, letters only
norm <- function(x) str_remove_all(str_to_lower(x), "[^a-z]")

# coded state names -> 2011 Census state codes (Telangana was part of Andhra Pradesh in 2011)
state_codes <- c(
  "andhrapradesh" = "28", "telangana" = "28", "bihar" = "10", "chhattisgarh" = "22",
  "jharkhand" = "20", "odisha" = "21", "westbengal" = "19", "maharashtra" = "27",
  "uttarpradesh" = "09", "karnataka" = "29", "kerala" = "32", "kerela" = "32",
  "madhyapradesh" = "23", "tamilnadu" = "33", "gujarat" = "24", "delhi" = "07",
  "newdelhi" = "07", "haryana" = "06", "punjab" = "03", "uttarakhand" = "05", "goa" = "30"
)


# 2011 names ---------------------------------------------------------------

shrug_names <- read_csv(names_file, show_col_types = FALSE) |>
  mutate(state_id       = str_sub(shrid2, 4, 5),
         district_id    = str_sub(shrid2, 7, 9),
         subdistrict_id = str_sub(shrid2, 11, 15))

# a few units carry two spellings in SHRUG; keep the first
districts_2011 <- shrug_names |>
  distinct(state_id, district_id, .keep_all = TRUE) |>
  select(state_id, district_id, state_name, district_name)

subdistricts_2011 <- shrug_names |>
  distinct(state_id, district_id, subdistrict_id, .keep_all = TRUE) |>
  select(state_id, district_id, subdistrict_id, subdistrict_name)


# Hand-entered pins -> SHRUG place -----------------------------------------

incidents <- read_csv(incident_file, show_col_types = FALSE) |>
  mutate(state_id = unname(state_codes[norm(state)]))

study_states <- sort(unique(na.omit(incidents$state_id)))
message("Reading SHRUG polygons for ", length(study_states), " states")

polys <- st_read(
  gpkg_file,
  query = sprintf("SELECT * FROM village WHERE pc11_state_id IN (%s)",
                  paste0("'", study_states, "'", collapse = ",")),
  quiet = TRUE
) |>
  rename(state_id = pc11_state_id, district_id = pc11_district_id,
         subdistrict_id = pc11_subdistrict_id, place_id = pc11_town_village_id,
         name = town_village_name)

pins <- incidents |>
  filter(!is.na(latitude), !is.na(longitude)) |>
  select(incident_number, state, district, latitude, longitude) |>
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326, remove = FALSE)

pin_places <- st_join(pins, select(polys, state_id, district_id, subdistrict_id, place_id, name),
                      join = st_intersects, left = TRUE) |>
  st_drop_geometry() |>
  distinct(incident_number, .keep_all = TRUE) |>
  rename(pin_state_id = state_id, pin_district_id = district_id,
         pin_subdistrict_id = subdistrict_id, pin_place_id = place_id, pin_place_name = name)

message("Pins: ", nrow(pins), "; inside a SHRUG polygon: ", sum(!is.na(pin_places$pin_place_id)))


# Coded district -> 2011 district ------------------------------------------

coded <- incidents |>
  filter(!is.na(state_id), !is.na(district)) |>
  count(state, district, state_id, name = "n_incidents")

# best 2011 name in the same state: exact, contained (Dantewada in Dakshin Bastar Dantewada),
# or closest by Jaro-Winkler distance
name_match <- function(sid, dist) {
  cand <- filter(districts_2011, state_id == sid)
  key  <- norm(dist)
  if (!nzchar(key)) return(tibble(name_district_id = NA_character_, name_dist = NA_real_))
  ck   <- norm(cand$district_name)
  d    <- stringdist(key, ck, method = "jw", p = 0.1)
  contained <- str_detect(ck, fixed(key)) | str_detect(key, fixed(ck))
  d[contained] <- pmin(d[contained], 0.05)
  d[ck == key] <- 0
  i <- which.min(d)
  tibble(name_district_id = cand$district_id[i], name_dist = round(d[i], 3))
}

# where the coded district's pins fall: the most common 2011 district and its share
pin_mode <- pin_places |>
  filter(!is.na(pin_district_id)) |>
  count(state, district, pin_state_id, pin_district_id) |>
  group_by(state, district) |>
  mutate(n_pins = sum(n), pin_share = n / n_pins) |>
  slice_max(n, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(state, district, pin_state_id, pin_district_id, pin_share, n_pins)

crosswalk <- coded |>
  bind_cols(purrr::map2_dfr(coded$state_id, coded$district, name_match)) |>
  left_join(pin_mode, by = c("state", "district")) |>
  mutate(
    pins_agree = !is.na(pin_district_id) & pin_state_id == state_id & pin_district_id == name_district_id,
    good_name  = name_dist <= 0.1,
    good_pins  = !is.na(pin_share) & pin_share >= 0.6 & n_pins >= 3,
    method = case_when(
      name_dist == 0        ~ "exact name",
      good_name & pins_agree ~ "name and pins",
      good_pins             ~ "pins",
      good_name             ~ "name",
      TRUE                  ~ NA_character_
    ),
    district_id_2011 = case_when(
      method == "pins"  ~ pin_district_id,
      !is.na(method)    ~ name_district_id
    ),
    review = is.na(method) | (good_name & good_pins & !pins_agree)
  )

# hand fixes, applied last
if (file.exists(override_file)) {
  overrides <- read_csv(override_file, col_types = cols(.default = "c")) |>
    select(state, district, state_id_override = state_id_2011, district_id_override = district_id_2011)
  crosswalk <- crosswalk |>
    left_join(overrides, by = c("state", "district")) |>
    mutate(method           = if_else(!is.na(district_id_override), "override", method),
           state_id         = coalesce(state_id_override, state_id),
           district_id_2011 = coalesce(district_id_override, district_id_2011),
           review           = review & is.na(district_id_override)) |>
    select(-state_id_override, -district_id_override)
}

crosswalk <- crosswalk |>
  left_join(rename(districts_2011, district_name_2011 = district_name) |> select(-state_name),
            by = c("state_id", "district_id_2011" = "district_id")) |>
  left_join(rename(districts_2011, name_candidate = district_name) |> select(-state_name),
            by = c("state_id", "name_district_id" = "district_id")) |>
  left_join(rename(districts_2011, pin_candidate = district_name) |> select(-state_name),
            by = c("pin_state_id" = "state_id", "pin_district_id" = "district_id")) |>
  select(state, district, n_incidents, state_id, district_id_2011, district_name_2011, method, review,
         name_candidate, name_dist, pin_candidate, pin_share, n_pins) |>
  arrange(desc(review), desc(n_incidents))

write_csv(crosswalk, file.path(out_dir, "district_crosswalk.csv"))
message("Crosswalk: ", nrow(crosswalk), " coded districts (", sum(crosswalk$n_incidents), " incidents); ",
        sum(crosswalk$review), " need review (", sum(crosswalk$n_incidents[crosswalk$review]), " incidents)")
print(count(crosswalk, method, wt = n_incidents, name = "incidents"))


# Places in the matched 2011 districts --------------------------------------

keep <- crosswalk |>
  filter(!is.na(district_id_2011)) |>
  distinct(state_id, district_id = district_id_2011)

places <- polys |>
  semi_join(keep, by = c("state_id", "district_id")) |>
  mutate(geom = st_point_on_surface(geom)) |>
  mutate(lon = st_coordinates(geom)[, 1], lat = st_coordinates(geom)[, 2]) |>
  st_drop_geometry() |>
  mutate(shrid2     = paste("11", state_id, district_id, subdistrict_id, place_id, sep = "-"),
         place_type = if_else(str_starts(place_id, "8"), "town", "village")) |>
  left_join(subdistricts_2011, by = c("state_id", "district_id", "subdistrict_id")) |>
  left_join(districts_2011, by = c("state_id", "district_id")) |>
  select(shrid2, state_id, district_id, subdistrict_id, place_id, place_type, name,
         subdistrict_name, district_name, state_name, lon, lat)

write_parquet(places, file.path(out_dir, "places.parquet"))
write_parquet(pin_places, file.path(out_dir, "pin_places.parquet"))
message("Places: ", nrow(places), " in ", nrow(keep), " districts")

# boundaries for the app's locate-by-coordinates box, simplified to about 50 m and stored as WKB
shapes <- polys |>
  semi_join(keep, by = c("state_id", "district_id")) |>
  mutate(geom = st_simplify(geom, preserveTopology = TRUE, dTolerance = 0.0005))

write_parquet(
  arrow_table(
    shrid2      = paste("11", shapes$state_id, shapes$district_id, shapes$subdistrict_id, shapes$place_id, sep = "-"),
    state_id    = shapes$state_id,
    district_id = shapes$district_id,
    wkb         = Array$create(unclass(st_as_binary(st_geometry(shapes))), type = binary())
  ),
  file.path(out_dir, "place_shapes.parquet")
)

# subdistrict boundaries in the study states: subdistricts cover all land, so a point outside every
# village (often reserved forest) still falls in one
subdistricts <- st_read(
  subdist_file,
  query = sprintf("SELECT * FROM subdistrict WHERE pc11_state_id IN (%s)",
                  paste0("'", study_states, "'", collapse = ",")),
  quiet = TRUE
) |>
  mutate(geom = st_simplify(geom, preserveTopology = TRUE, dTolerance = 0.0005))

write_parquet(
  arrow_table(
    state_id       = subdistricts$pc11_state_id,
    district_id    = subdistricts$pc11_district_id,
    subdistrict_id = subdistricts$pc11_subdistrict_id,
    wkb            = Array$create(unclass(st_as_binary(st_geometry(subdistricts))), type = binary())
  ),
  file.path(out_dir, "subdistrict_shapes.parquet")
)
message("Subdistricts: ", nrow(subdistricts), " in ", length(study_states), " states")
