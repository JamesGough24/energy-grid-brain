"""
data_ingestion.py

Downloads Ontario grid data from IESO's public reports site, entirely scoped to just
the legacy market era (2002 - April 30, 2025):

  - Hourly Ontario/Market Demand: https://reports-public.ieso.ca/public/Demand/
  - Legacy HOEP (Hourly Ontario Energy Price): https://reports-public.ieso.ca/public/PriceHOEPPredispOR/

Both are published as static yearly bulk CSV files (not day-by-day reports),
so they are NOT subject to IESO's 90-day rolling retention policy that
affects the newer post-MRP reports like DA-OZP
"""

import io
import time
import logging
from pathlib import Path
from datetime import date

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://reports-public.ieso.ca/public"
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HOEP_RETIREMENT_DATE = date(2025, 4, 30)  # Last day HOEP was published

HEADERS = {"User-Agent": "ontario-grid-brain-student-project/1.0"}
REQUEST_DELAY_SECONDS = 0.3  # Add delay to not get rate limited


# ---------------------------------------------------------------------------
# 1. Demand (2002-present, single continuous series)
# ---------------------------------------------------------------------------

def fetch_demand_history(start_year: int = 2002, end_year: int = 2025) -> pd.DataFrame:
    """Downloads yearly Demand CSVs and concatenates them into one DataFrame."""
    frames = []

    for year in range(start_year, end_year + 1):
        url = f"{BASE_URL}/Demand/PUB_Demand_{year}.csv"
        cache_path = RAW_DIR / f"demand_{year}.csv"
        df = _get_csv_cached(url, cache_path, skiprows=3)
        if df is not None:
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True).drop_duplicates()
    combined.to_csv(RAW_DIR / "demand_full_history.csv", index=False)
    log.info("Demand history: %d rows across %d years", len(combined), len(frames))
    return combined


# ---------------------------------------------------------------------------
# 2. Legacy HOEP (2002 - April 30, 2025)
# ---------------------------------------------------------------------------

def fetch_hoep_history(start_year: int = 2002, end_year: int = 2025) -> pd.DataFrame:
    """Downloads yearly legacy HOEP CSVs."""
    frames = []

    for year in range(start_year, end_year + 1):
        url = f"{BASE_URL}/PriceHOEPPredispOR/PUB_PriceHOEPPredispOR_{year}.csv"
        cache_path = RAW_DIR / f"hoep_{year}.csv"
        df = _get_csv_cached(url, cache_path, skiprows=3)
        if df is not None:
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True).drop_duplicates()

    combined.to_csv(RAW_DIR / "hoep_full_history.csv", index=False)
    log.info("Legacy HOEP history: %d rows across %d years", len(combined), len(frames))
    return combined


# ---------------------------------------------------------------------------
# 3. Merge demand + price into one modeling dataset
# ---------------------------------------------------------------------------

def build_combined_dataset() -> pd.DataFrame:
    """
    Merges Demand and HOEP into a single hourly DataFrame covering
    2002 - April 30, 2025. This is the full dataset the rest of the
    pipeline (feature engineering, forecasting, classification,
    optimization, simulation) will be built on.
    """
    demand = fetch_demand_history()
    hoep = fetch_hoep_history()

    demand = demand.rename(columns={"Market Demand": "market_demand", "Ontario Demand": "ontario_demand"})
    hoep = hoep.rename(columns={"HOEP": "hoep"})

    # Combine date and hour of each entry to form a datetime we can use
    demand["datetime"] = pd.to_datetime(demand["Date"]) + pd.to_timedelta(demand["Hour"].astype(int) - 1, unit="h")
    hoep["datetime"] = pd.to_datetime(hoep["Date"]) + pd.to_timedelta(hoep["Hour"].astype(int) - 1, unit="h")

    merged = demand.merge(hoep[["datetime", "hoep"]], on="datetime", how="inner")
    merged = merged.sort_values("datetime").drop_duplicates(subset="datetime")

    out_path = RAW_DIR / "combined_dataset.csv"
    merged.to_csv(out_path, index=False)
    log.info("Combined dataset: %d rows, %s to %s, saved to %s",
              len(merged), merged["datetime"].min(), merged["datetime"].max(), out_path)
    return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_csv_cached(url: str, cache_path: Path, skiprows: int = 0) -> pd.DataFrame | None:
    """Downloads a CSV (with local caching) and returns it as a DataFrame, or None on failure."""
    if cache_path.exists():
        return pd.read_csv(cache_path, skiprows=skiprows)

    # Check the response status code of the GET request then save the data in a CSV file in src/data/raw if all good
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        return pd.read_csv(io.BytesIO(resp.content), skiprows=skiprows)
    except requests.exceptions.RequestException as e:
        log.warning("Could not fetch %s: %s", url, e)
        return None
    finally:
        time.sleep(REQUEST_DELAY_SECONDS)


if __name__ == "__main__":
    build_combined_dataset()
