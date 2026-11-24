"""Build the Helsinki-region hexagon dataset used in the week-10 GeoAI tutorial.

What it does
------------
1. Downloads points of interest (POIs) and land-use polygons from OpenStreetMap
   for the Helsinki capital region (Helsinki, Espoo, Vantaa, Kauniainen).
2. Lays an H3 hexagon grid over the four municipalities.
3. Counts POIs per hexagon in a handful of readable categories  -> the FEATURES.
4. Derives a land-use class (green / residential / work) per hexagon -> the LABEL.
5. Writes data/helsinki_hex_r<resolution>.gpkg (one row per hexagon) and data/pois_r<resolution>.csv.gz.

Features and label come from different OSM layers on purpose: the features are
things people mapped as points of interest, the label is how the ground is zoned.

Run:  python build_dataset.py 8          (H3 resolution 8 or 9; takes a few minutes, needs internet)
Data: (c) OpenStreetMap contributors, Open Database License (ODbL).
"""
import sys
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

PLACES = ["Helsinki, Finland", "Espoo, Finland", "Vantaa, Finland", "Kauniainen, Finland"]
H3_RES = int(sys.argv[1]) if len(sys.argv) > 1 else 8   # 8: about 0.57 km2 per hexagon in this region; 9: about 0.08 km2
METRIC_CRS = "EPSG:3067"   # ETRS-TM35FIN, metres
MIN_COVER = 0.20           # a hexagon needs >= 20 % classified land use to be kept (drops the sea)
BUILT_MIN = 0.20           # >= 20 % built-up land makes a hexagon "residential" or "work"

ox.settings.use_cache = True
ox.settings.cache_folder = str(DATA / "_osm_cache")
ox.settings.requests_timeout = 600


# --------------------------------------------------------------------------- h3 v3 / v4 compatibility
def cells_in_polygon(poly, res):
    """H3 cells whose centre falls inside a shapely polygon (lon/lat)."""
    if hasattr(h3, "polyfill"):  # h3 v3
        return h3.polyfill(poly.__geo_interface__, res, geo_json_conformant=True)
    return h3.geo_to_cells(poly, res)  # h3 v4


def cell_polygon(cell):
    if hasattr(h3, "h3_to_geo_boundary"):  # h3 v3
        return Polygon(h3.h3_to_geo_boundary(cell, geo_json=True))
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])  # h3 v4


# --------------------------------------------------------------------------- POI categories (features)
FOOD = {"restaurant", "cafe", "fast_food", "bar", "pub", "food_court", "ice_cream", "biergarten", "nightclub"}
EDUCATION = {"school", "kindergarten", "university", "college", "library", "childcare", "music_school"}
HEALTH = {"hospital", "clinic", "pharmacy", "doctors", "dentist", "veterinary", "social_facility", "nursing_home"}
CAR = {"parking", "parking_entrance", "fuel", "charging_station", "car_wash", "car_rental", "car_sharing"}
BIKE = {"bicycle_parking", "bicycle_rental", "bicycle_repair_station"}
CIVIC = {"townhall", "police", "fire_station", "post_office", "community_centre", "place_of_worship",
         "courthouse", "embassy", "post_box", "parcel_locker", "public_building"}
CULTURE = {"theatre", "cinema", "arts_centre", "museum", "events_venue", "conference_centre"}
FINANCE = {"bank", "atm", "bureau_de_change"}
STREET = {"bench", "waste_basket", "recycling", "shelter", "toilets", "drinking_water", "waste_disposal",
          "vending_machine", "bbq", "fountain", "clock", "telephone"}
SPORT = {"pitch", "sports_centre", "fitness_centre", "fitness_station", "swimming_pool", "track",
         "stadium", "ice_rink", "sports_hall", "golf_course", "horse_riding", "marina", "slipway"}
PLAY = {"playground", "dog_park", "picnic_table", "firepit", "bird_hide", "sauna"}
DAILY_SHOPS = {"supermarket", "convenience", "kiosk", "bakery", "alcohol", "greengrocer", "butcher",
               "department_store", "mall", "general", "variety_store", "chemist"}
FEATURES = ["food", "shop_daily", "shop_other", "education", "health", "car", "bike", "civic",
            "culture", "finance", "street_furniture", "sport", "play", "office", "lodging", "craft_industry"]


def poi_category(row):
    a, s, l, t = row.get("amenity"), row.get("shop"), row.get("leisure"), row.get("tourism")
    if isinstance(s, str):
        return "shop_daily" if s in DAILY_SHOPS else "shop_other"
    if isinstance(a, str):
        for name, group in [("food", FOOD), ("education", EDUCATION), ("health", HEALTH), ("car", CAR),
                            ("bike", BIKE), ("civic", CIVIC), ("culture", CULTURE), ("finance", FINANCE),
                            ("street_furniture", STREET)]:
            if a in group:
                return name
        return None
    if isinstance(l, str):
        return "sport" if l in SPORT else ("play" if l in PLAY else None)   # parks are NOT a feature (label leak)
    if isinstance(t, str):
        if t in {"hotel", "hostel", "guest_house", "apartment", "motel"}:
            return "lodging"
        return "culture" if t in {"museum", "gallery", "attraction", "artwork", "viewpoint"} else None
    if isinstance(row.get("office"), str):
        return "office"
    if isinstance(row.get("craft"), str) or isinstance(row.get("man_made"), str):
        return "craft_industry"
    return None


def landmark_rank(row):
    """How recognisable a POI is as a landmark for its hexagon (0 = best, 9 = do not use)."""
    a, s, l, t = row.get("amenity"), row.get("shop"), row.get("leisure"), row.get("tourism")
    if a in {"university", "college"}:
        return 0
    if s in {"mall", "department_store"}:
        return 1
    if a in {"library", "theatre", "cinema", "hospital", "townhall"} or t == "museum":
        return 2
    if l in {"sports_centre", "stadium", "ice_rink", "golf_course", "marina"}:
        return 3
    if a == "school":
        return 4
    if s == "supermarket" or t == "hotel":
        return 5
    if a in {"community_centre", "place_of_worship", "kindergarten"}:
        return 6
    if a in {"restaurant", "cafe"}:
        return 7
    return 9


# --------------------------------------------------------------------------- land-use classes (label)
def landuse_class(row):
    lu, le, na = row.get("landuse"), row.get("leisure"), row.get("natural")
    if lu == "residential":
        return "residential"
    if lu in {"commercial", "retail"}:
        return "commercial"
    if lu in {"industrial", "railway", "port", "quarry", "landfill", "depot"}:
        return "industrial"
    if lu in {"forest", "meadow", "grass", "recreation_ground", "cemetery", "farmland", "allotments",
              "village_green", "orchard", "greenfield"}:
        return "green"
    if le in {"park", "nature_reserve", "garden", "golf_course"}:
        return "green"
    if na in {"wood", "scrub", "heath", "grassland", "wetland"}:
        return "green"
    return None


def main():
    # 1. study area ---------------------------------------------------------------------------------
    area = ox.geocode_to_gdf(PLACES)[["geometry"]]
    area["municipality"] = [p.split(",")[0] for p in PLACES]
    region = area.geometry.union_all() if hasattr(area.geometry, "union_all") else area.unary_union
    west, south, east, north = region.bounds
    print(f"study area bbox: {west:.3f} {south:.3f} {east:.3f} {north:.3f}")

    # 2. hexagons -----------------------------------------------------------------------------------
    cells = set()
    for geom in area.geometry:
        for part in getattr(geom, "geoms", [geom]):
            cells |= set(cells_in_polygon(part, H3_RES))
    hexes = gpd.GeoDataFrame({"hex_id": sorted(cells)},
                             geometry=[cell_polygon(c) for c in sorted(cells)], crs="EPSG:4326")
    print(f"hexagons in the four municipalities (incl. sea): {len(hexes)}")

    # 3. POIs -> features ---------------------------------------------------------------------------
    poi_tags = {"amenity": True, "shop": True, "leisure": True, "tourism": True, "office": True,
                "craft": True, "man_made": ["works", "wastewater_plant", "storage_tank", "silo"]}
    raw = ox.features_from_bbox(bbox=(west, south, east, north), tags=poi_tags)
    print(f"raw OSM POI features: {len(raw)}")
    raw = raw.reset_index()
    raw["category"] = raw.apply(poi_category, axis=1)
    raw["fame"] = raw.apply(landmark_rank, axis=1)
    pois = raw.dropna(subset=["category"]).copy()
    pois["geometry"] = pois.geometry.to_crs(METRIC_CRS).centroid.to_crs("EPSG:4326")  # polygons -> points
    pois = gpd.GeoDataFrame(pois[["category", "name", "fame", "geometry"]], crs="EPSG:4326")
    pois = gpd.sjoin(pois, hexes, predicate="within").drop(columns="index_right")
    counts = (pois.groupby(["hex_id", "category"]).size().unstack(fill_value=0)
              .reindex(columns=FEATURES, fill_value=0))
    print("POIs kept, by category:\n", counts.sum().sort_values(ascending=False).to_string())

    # 4. land use -> label --------------------------------------------------------------------------
    lu_tags = {"landuse": True, "leisure": ["park", "nature_reserve", "garden", "golf_course"],
               "natural": ["wood", "scrub", "heath", "grassland", "wetland"]}
    lu = ox.features_from_bbox(bbox=(west, south, east, north), tags=lu_tags).reset_index()
    lu = lu[lu.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    lu["lu_class"] = lu.apply(landuse_class, axis=1)
    lu = lu.dropna(subset=["lu_class"])[["lu_class", "geometry"]].to_crs(METRIC_CRS)
    lu["geometry"] = lu.geometry.buffer(0)
    print(f"classified land-use polygons: {len(lu)}")

    hex_m = hexes.to_crs(METRIC_CRS)
    hex_m["hex_area"] = hex_m.area
    classes = ["residential", "commercial", "industrial", "green"]
    hex_area = hex_m.set_index("hex_id")["hex_area"]
    share = pd.DataFrame(0.0, index=hex_m["hex_id"], columns=classes)
    for c in classes:
        # cut the polygons of this class by the hexagons, then merge the pieces inside each hexagon,
        # so that overlapping polygons of one class are not counted twice
        inter = gpd.overlay(hex_m[["hex_id", "geometry"]], lu.loc[lu.lu_class == c, ["geometry"]],
                            how="intersection", keep_geom_type=True)
        inter_area = inter.dissolve("hex_id").area
        share.loc[inter_area.index, c] = (inter_area / hex_area.loc[inter_area.index]).values
    share.columns = [f"share_{c}" for c in classes]

    # 5. assemble -----------------------------------------------------------------------------------
    out = hexes.merge(counts, left_on="hex_id", right_index=True, how="left")
    out[FEATURES] = out[FEATURES].fillna(0).astype(int)
    out = out.merge(share, left_on="hex_id", right_index=True, how="left")
    share_cols = list(share.columns)
    out["cover"] = out[share_cols].sum(axis=1).clip(upper=1.0)
    out["poi_total"] = out[FEATURES].sum(axis=1)
    # Label rule. Taking the largest class would call 3 hexagons in 4 "green", because forest polygons are
    # huge. Instead: a hexagon with at least BUILT_MIN built-up land is "residential" or "work" (commercial,
    # retail, industrial), whichever covers more; everything else is "green".
    out["share_work"] = out["share_commercial"] + out["share_industrial"]
    built = out["share_residential"] + out["share_work"]
    out["landuse"] = np.where(built < BUILT_MIN, "green",
                              np.where(out["share_work"] >= out["share_residential"], "work", "residential"))
    out = out[out["cover"] >= MIN_COVER].copy()            # drops sea and unmapped hexagons
    cent = out.geometry.to_crs(METRIC_CRS).centroid
    out["x"], out["y"] = cent.x.round(1), cent.y.round(1)
    out = gpd.sjoin(out, area, predicate="intersects").drop(columns="index_right").drop_duplicates("hex_id")
    out = out.sort_values("hex_id").reset_index(drop=True)

    print(f"\nhexagons kept: {len(out)}")
    print("label counts:\n", out["landuse"].value_counts().to_string())
    print("hexagons without any POI:", int((out.poi_total == 0).sum()))

    # 6. two recognisable named POIs per hexagon, so that a hexagon can be recognised on a slide ---------
    #    (taken from the POIs already downloaded; no extra request to the OSM servers)
    named = pois.dropna(subset=["name"]).copy()
    named = named[(named["fame"] < 9) & (named["name"].str.len() > 3)]
    named = named.sort_values(["hex_id", "fame", "name"]).drop_duplicates(["hex_id", "name"])
    landmarks = named.groupby("hex_id")["name"].apply(lambda s: "; ".join(s.head(2)))
    out["landmarks"] = out["hex_id"].map(landmarks).fillna("")

    hex_file = DATA / f"helsinki_hex_r{H3_RES}.gpkg"
    out.to_file(hex_file, driver="GPKG")
    keep = pois[pois.hex_id.isin(out.hex_id)].copy()
    keep["lon"], keep["lat"] = keep.geometry.x.round(6), keep.geometry.y.round(6)
    keep[["category", "name", "lon", "lat", "hex_id"]].to_csv(DATA / f"pois_r{H3_RES}.csv.gz", index=False)
    print("wrote", hex_file)


if __name__ == "__main__":
    main()
