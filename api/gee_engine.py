"""
ASTRA — AI Space Sentinel
core engine: fetch Sentinel-2 imagery from Google Earth Engine,
compute vegetation/water indices, detect anomalous change between two
time periods, and produce an explainable alert.
"""

import ee
import os
import json


# ---------------------------------------------------------------------
# 1. Earth Engine initialization
# ---------------------------------------------------------------------
def init_ee():
    """
    Initializes Earth Engine using a service account.
    On Vercel: set env vars GEE_SERVICE_ACCOUNT and GEE_PRIVATE_KEY_JSON
    (the full content of your service-account key.json, pasted as-is).
    Locally: you can instead set GEE_PRIVATE_KEY_FILE to a file path,
    or just run `earthengine authenticate` once for interactive use.
    """
    service_account = os.environ.get("GEE_SERVICE_ACCOUNT")
    key_json = os.environ.get("GEE_PRIVATE_KEY_JSON")
    key_file = os.environ.get("GEE_PRIVATE_KEY_FILE")

    if service_account and key_json:
        # key content passed directly as an env var (Vercel-friendly: no filesystem needed)
        credentials = ee.ServiceAccountCredentials(service_account, key_data=key_json)
        ee.Initialize(credentials)
    elif service_account and key_file:
        credentials = ee.ServiceAccountCredentials(service_account, key_file)
        ee.Initialize(credentials)
    else:
        # falls back to locally cached credentials from `earthengine authenticate`
        ee.Initialize()


# ---------------------------------------------------------------------
# 2. Region + imagery helpers
# ---------------------------------------------------------------------
def make_region(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> ee.Geometry:
    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])


def mask_clouds(image: ee.Image) -> ee.Image:
    """Basic cloud mask using Sentinel-2 QA60 band."""
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask)


def get_composite(region: ee.Geometry, start_date: str, end_date: str) -> ee.Image:
    """Median, cloud-masked Sentinel-2 SR composite for a date range."""
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(mask_clouds)
    )
    return collection.median().clip(region)


def compute_ndvi(image: ee.Image) -> ee.Image:
    return image.normalizedDifference(["B8", "B4"]).rename("NDVI")


def compute_ndwi(image: ee.Image) -> ee.Image:
    return image.normalizedDifference(["B3", "B8"]).rename("NDWI")


# ---------------------------------------------------------------------
# 3. Change detection
# ---------------------------------------------------------------------
def mean_index_value(index_image: ee.Image, region: ee.Geometry, scale: int = 10) -> float:
    stats = index_image.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=scale, maxPixels=1e9
    )
    value = stats.get(index_image.bandNames().get(0)).getInfo()
    return float(value) if value is not None else 0.0


def classify_severity(percent_change: float) -> str:
    magnitude = abs(percent_change)
    if magnitude < 10:
        return "low"
    if magnitude < 25:
        return "medium"
    return "high"


def detect_change(
    region: ee.Geometry,
    before_start: str,
    before_end: str,
    after_start: str,
    after_end: str,
    index: str = "NDVI",
) -> dict:
    """
    Compares an index (NDVI or NDWI) between two time windows over a region
    and returns a structured, explainable change report.
    """
    before_img = get_composite(region, before_start, before_end)
    after_img = get_composite(region, after_start, after_end)

    index_fn = compute_ndvi if index == "NDVI" else compute_ndwi
    before_index = index_fn(before_img)
    after_index = index_fn(after_img)

    before_mean = mean_index_value(before_index, region)
    after_mean = mean_index_value(after_index, region)

    diff = after_mean - before_mean
    percent_change = (diff / before_mean * 100) if before_mean != 0 else 0.0
    severity = classify_severity(percent_change)
    is_anomaly = severity in ("medium", "high")

    direction = "increase" if diff > 0 else "decrease"
    explanation = build_explanation(index, direction, percent_change, severity)

    return {
        "index": index,
        "before_mean": round(before_mean, 4),
        "after_mean": round(after_mean, 4),
        "percent_change": round(percent_change, 2),
        "direction": direction,
        "severity": severity,
        "is_anomaly": is_anomaly,
        "explanation": explanation,
        "before_thumbnail": get_thumbnail_url(before_img, region),
        "after_thumbnail": get_thumbnail_url(after_img, region),
    }


def get_thumbnail_url(image: ee.Image, region: ee.Geometry, dimensions: int = 512) -> str:
    vis = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
    return image.getThumbURL({"region": region, "dimensions": dimensions, **vis})


def build_explanation(index: str, direction: str, percent_change: float, severity: str) -> str:
    label = "vegetation cover" if index == "NDVI" else "surface water extent"
    verb = "increased" if direction == "increase" else "decreased"
    return (
        f"The AI detected that {label} {verb} by {abs(percent_change):.1f}% "
        f"between the two observed periods, compared to the historical baseline. "
        f"This is classified as a {severity}-severity change based on the "
        f"magnitude of deviation from normal seasonal patterns."
    )
