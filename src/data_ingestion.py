"""
Downloads Ontario grid data from IESO's public reports site, entirely scoped to just
the legacy market era (2002 - April 30, 2025):

  - Hourly Ontario/Market Demand: https://reports-public.ieso.ca/public/Demand/
  - Legacy HOEP (Hourly Ontario Energy Price): https://reports-public.ieso.ca/public/PriceHOEPPredispOR/

Both are published as static yearly bulk CSV files (not day-by-day reports),
so they are NOT subject to IESO's 90-day rolling retention policy that
affects the newer post-MRP reports like DA-OZP

Also downloads hourly temperature data from Environment Canada, scoped to the same timeframe as
HOEP data (2002 - April 30, 2025). Weather station used will be Toronto Pearson Airport. Although
weather will vary massively across the province, using Toronto Pearson will cover the temperature
in the province's biggest power usage hub, while still being a decent indicator of conditions 
elsewhere.

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
WEATHER_BASE_URL = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
RAW_DIR = Path(__file__).resolve().parent.parent / "data/raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HOEP_RETIREMENT_DATE = date(2025, 4, 30)  # Last day HOEP was published

# Station ID for Toronto Pearson Airport, acting as a proxy to represent temperature in Ontario.
WEATHER_STATION_ID_FIRST = 5097
WEATHER_STATION_ID_SECOND = 51459

HEADERS = {"User-Agent": "ontario-grid-brain-student-project/1.0"}
REQUEST_DELAY_SECONDS = 0.3  # delay between IESO requests
WEATHER_REQUEST_DELAY_SECONDS = 2.0  # delay between ECCC requests
MAX_RETRIES = 4
RETRY_BACKOFF_BASE = 3.0  # seconds; doubles each retry (3s, 6s, 12s, 24s)

# ---------------------------------------------------------------------------
# Demand (2002-present, single continuous series)
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
# Legacy HOEP (May 1, 2002 - April 30, 2025)
# ---------------------------------------------------------------------------

def fetch_hoep_history(start_year: int = 2002, end_year: int = 2025) -> pd.DataFrame:
    """Downloads yearly legacy HOEP CSVs"""
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
# Weather Data (May 1, 2002 - April 30, 2025)
# ---------------------------------------------------------------------------

def fetch_weather_history(start_year: int = 2002, end_year: int = 2025, station_id_first: int = WEATHER_STATION_ID_FIRST, station_id_second = WEATHER_STATION_ID_SECOND) -> pd.DataFrame:
    month_cache_dir = RAW_DIR / "weather_monthly"
    month_cache_dir.mkdir(exist_ok=True)

    frames = []
    failed_months = []

    def _fetch_month(year: int, month: int, station_id: int):
        cache_path = month_cache_dir / f"weather_{year}_{month:02d}.csv"
        df = _get_csv_cached(
            f"{WEATHER_BASE_URL}?format=csv&stationID={station_id}"
            f"&Year={year}&Month={month}&Day=1&timeframe=1&submit=Download+Data",
            cache_path,
            skiprows=0,
            delay=WEATHER_REQUEST_DELAY_SECONDS
        )
        if df is not None:
            frames.append(df)
        else:
            failed_months.append((year, month, station_id))

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == 2002 and month < 5:
                continue # HOEP data started on May 1, 2002
            if year == 2025 and month > 4:
                break # to not go past retirement date of HOEP

            if year == 2013 and month == 6:
                cache_path_station1 = month_cache_dir / f"{station_id_first}_weather_{year}_{month:02d}.csv"
                station1 = _get_csv_cached(
                    f"{WEATHER_BASE_URL}?format=csv&stationID={station_id_first}"
                    f"&Year={year}&Month={month}&Day=1&timeframe=1&submit=Download+Data",
                    cache_path_station1,
                    skiprows=0,
                    delay=WEATHER_REQUEST_DELAY_SECONDS
                )

                cache_path_station2 = month_cache_dir / f"{station_id_second}_weather_{year}_{month:02d}.csv"
                station2 = _get_csv_cached(
                    f"{WEATHER_BASE_URL}?format=csv&stationID={station_id_second}"
                    f"&Year={year}&Month={month}&Day=1&timeframe=1&submit=Download+Data",
                    cache_path_station2,
                    skiprows=0,
                    delay=WEATHER_REQUEST_DELAY_SECONDS,
                )

                final_cache_path = month_cache_dir / f"weather_{year}_{month:02d}.csv"

                if station1 is not None and station2 is not None:
                    station1 = station1.set_index("Date/Time (LST)")
                    station2 = station2.set_index("Date/Time (LST)")
                    station1 = station1.reindex(columns=station2.columns)

                    station1["Temp (\u00b0C)"] = pd.to_numeric(station1["Temp (\u00b0C)"], errors="coerce")
                    station2["Temp (\u00b0C)"] = pd.to_numeric(station2["Temp (\u00b0C)"], errors="coerce")

                    station1["Temp (\u00b0C)"] = station1["Temp (\u00b0C)"].fillna(station2["Temp (\u00b0C)"])

                    merged_june = station1.reset_index()
                    merged_june = merged_june.drop_duplicates(subset=["Date/Time (LST)"], keep="first")

                    # Convert file to CSV then remove the 2 individual station data files 
                    merged_june.to_csv(final_cache_path, index=False)
                    cache_path_station1.unlink(missing_ok=True)
                    cache_path_station2.unlink(missing_ok=True)
                    frames.append(merged_june)

                elif station1 is not None:
                    station1.to_csv(final_cache_path, index=False)
                    cache_path_station1.unlink(missing_ok=True)
                    cache_path_station2.unlink(missing_ok=True)
                    frames.append(station1)

                elif station2 is not None:
                    station2.to_csv(final_cache_path, index=False)
                    cache_path_station1.unlink(missing_ok=True)
                    cache_path_station2.unlink(missing_ok=True)
                    frames.append(station2)

                continue

            # Up to and including June 2013 using first station ID
            if (year < 2013) or (year == 2013 and month <= 6):
                _fetch_month(year, month, station_id_first)

            # From July 2013 onwards using second station ID
            if (year > 2013) or (year == 2013 and month >= 6):
                _fetch_month(year, month, station_id_second)

    while failed_months:
        cooldown = 15
        log.warning(
            "%d month(s) failed after all per-request retries: %s. "
            "Cooling down %ds, then retrying them once more automatically "
            "before giving up.",
            len(failed_months), failed_months, cooldown,
        )
        time.sleep(cooldown)

        for year, month in failed_months:
            cache_path = month_cache_dir / f"weather_{year}_{month:02d}.csv"
            if year < 2013 or (year == 2013 and month < 6):
                station_id = station_id_first
            else:
                station_id = station_id_second
            df = _get_csv_cached(
                f"{WEATHER_BASE_URL}?format=csv&stationID={station_id}"
                f"&Year={year}&Month={month}&Day=1&timeframe=1&submit=Download+Data",
                cache_path,
                skiprows=0,
                delay=WEATHER_REQUEST_DELAY_SECONDS,
            )
            if df is not None:
                frames.append(df)
                failed_months.remove((year, month))

         
    combined = pd.concat(frames, ignore_index=True).drop_duplicates()
    combined.to_csv(RAW_DIR / "weather_full_history.csv", index=False)
    log.info("Weather history: %d rows", len(combined))

    return combined

# ---------------------------------------------------------------------------
# Merge demand + price into one modeling dataset
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
    weather = fetch_weather_history()

    demand = demand.rename(columns={"Market Demand": "market_demand", "Ontario Demand": "ontario_demand"})
    hoep = hoep.rename(columns={"HOEP": "hoep"})
    print("Successfully renamed demand and hoep")
    weather = weather.rename(columns={"Date/Time (LST)": "datetime", "Temp (\u00b0C)": "temperature_c"})
    print("Successfully renamed weather")

    # Combine date and hour of each entry to form a datetime we can use
    demand["datetime"] = pd.to_datetime(demand["Date"]) + pd.to_timedelta(demand["Hour"].astype(int) - 1, unit="h")
    hoep["datetime"] = pd.to_datetime(hoep["Date"]) + pd.to_timedelta(hoep["Hour"].astype(int) - 1, unit="h")
    print("Successfully redid datetime for demand and hoep")
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    print("Successfully redid datetime for weather")

    merged = demand.merge(hoep[["datetime", "hoep"]], on="datetime", how="inner")
    print("Successfully merged demand and hoep")
    merged = merged.merge(weather[["datetime", "temperature_c"]], on="datetime", how="left")
    print("Successfully rmerged with weather")
    merged = merged.sort_values("datetime").drop_duplicates(subset="datetime")
    print("Successfully sorted values")

    out_path = RAW_DIR / "combined_dataset.csv"
    merged.to_csv(out_path, index=False)
    log.info("Combined dataset: %d rows, %s to %s, saved to %s",
              len(merged), merged["datetime"].min(), merged["datetime"].max(), out_path)
    return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_csv_cached(url: str, cache_path: Path, skiprows: int = 0, delay = REQUEST_DELAY_SECONDS) -> pd.DataFrame | None:
    """Downloads a CSV (with local caching) and returns it as a DataFrame, or None on failure."""
    if cache_path.exists():
        return pd.read_csv(cache_path, skiprows=skiprows)

    # Check the response status code of the GET request then save the data in a CSV file in src/data/raw if all good
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            cache_path.write_bytes(resp.content)
            time.sleep(delay)
            return pd.read_csv(io.BytesIO(resp.content), skiprows=skiprows)
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES:
                backoff = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                log.warning(
                    "Attempt %d/%d failed for %s (%s) -- retrying in %.1fs",
                    attempt, MAX_RETRIES, url, e, backoff,
                )
                time.sleep(backoff)
            else:
                log.warning(
                    "Giving up on %s after %d attempts (%s). Re-running this "
                    "function later will retry only this file, since "
                    "everything else is already cached.",
                    url, MAX_RETRIES, e,
                )
                time.sleep(delay)
                return None


if __name__ == "__main__":
    build_combined_dataset()
