"""
data_ingestion.py

Downloads and assembles Ontario grid data from IESO's public reports site:
  - Hourly Ontario/Market Demand (2002 - present)
  - Legacy HOEP (2002 - April 30, 2025)
  - Day-Ahead Ontario Zonal Price / DA-OZP (May 3, 2025 - present) -- one XML file per day
  - Load Forecast Deviation Adjustment / LFDA (May 2025 - present) -- bulk CSV, like Demand

All reports are read anonymously with no API key from IESO's public flat-file
repository: https://reports-public.ieso.ca/public/<ReportName>/
"""

import io
import time
import logging
from pathlib import Path
from datetime import date, timedelta

import requests
import pandas as pd
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://reports-public.ieso.ca/public"
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

MRP_LAUNCH_DATE = date(2025, 5, 1)          # Market Renewal Program went live on May 1, 2025
DA_OZP_FIRST_AVAILABLE = date(2025, 5, 3)   # earliest DA-OZP file found in the archive

HEADERS = {"User-Agent": "ontario-grid-brain-student-project/1.0"}
REQUEST_DELAY_SECONDS = 0.3          # add delay to avoid rate limiting


# ---------------------------------------------------------------------------
# 1. Demand (single continuous series, 2002-present, no market-change issues)
# ---------------------------------------------------------------------------

def fetch_demand_history(start_year: int = 2002, end_year: int | None = None) -> pd.DataFrame:
    """Downloads yearly Demand CSVs and concatenates them into one DataFrame."""
    end_year = end_year or date.today().year
    frames = []

    for year in range(start_year, end_year + 1):
        url = f"{BASE_URL}/Demand/PUB_Demand_{year}.csv"
        cache_path = RAW_DIR / f"demand_{year}.csv"

        df = _get_csv_cached(url, cache_path, skiprows=3)  # IESO CSVs have a header preamble
        if df is not None:
            df["year_source"] = year
            frames.append(df)

    # Also grab the current, still-accumulating year via the "most recent" file
    current_url = f"{BASE_URL}/Demand/PUB_Demand.csv"
    current_df = _get_csv_cached(current_url, RAW_DIR / "demand_current.csv", skiprows=3)
    if current_df is not None:
        frames.append(current_df)

    combined = pd.concat(frames, ignore_index=True).drop_duplicates()
    combined.to_csv(RAW_DIR / "demand_full_history.csv", index=False)
    log.info("Demand history: %d rows", len(combined))
    return combined


# ---------------------------------------------------------------------------
# 2. Legacy HOEP (2002 - April 30, 2025)
# ---------------------------------------------------------------------------

def fetch_hoep_history(start_year: int = 2002, end_year: int = 2025) -> pd.DataFrame:
    """Downloads yearly legacy HOEP CSVs. Report is archived after April 30, 2025."""
    frames = []

    for year in range(start_year, end_year + 1):
        url = f"{BASE_URL}/PriceHOEPPredispOR/PUB_PriceHOEPPredispOR_{year}.csv"
        cache_path = RAW_DIR / f"hoep_{year}.csv"
        df = _get_csv_cached(url, cache_path, skiprows=3)
        if df is not None:
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True).drop_duplicates()
    # Keep only rows up to the actual retirement date, in case the final year file
    # runs a few days past April 30, 2025
    if "Date" in combined.columns:
        combined["Date"] = pd.to_datetime(combined["Date"])
        combined = combined[combined["Date"] < pd.Timestamp(MRP_LAUNCH_DATE)]

    combined.to_csv(RAW_DIR / "hoep_full_history.csv", index=False)
    log.info("Legacy HOEP history: %d rows", len(combined))
    return combined


# ---------------------------------------------------------------------------
# 3. DA-OZP (May 3, 2025-present) -- ONE FILE PER DAY, must loop
# ---------------------------------------------------------------------------

def fetch_da_ozp_history(
    start_date: date = DA_OZP_FIRST_AVAILABLE,
    end_date: date | None = None,
) -> pd.DataFrame:
    """
    Loops day-by-day and downloads each daily DA-OZP XML file, parsing the
    24 hourly prices out of each one. This is the report that's structured
    differently from everything else -- there is no bulk/yearly file.
    """
    end_date = end_date or (date.today() - timedelta(days=1))  # yesterday is safest (today may not be published yet)
    day_cache_dir = RAW_DIR / "da_ozp_daily"
    day_cache_dir.mkdir(exist_ok=True)

    all_rows = []
    current = start_date
    n_days = (end_date - start_date).days + 1
    log.info("Fetching DA-OZP for %d days (%s to %s)", n_days, start_date, end_date)

    while current <= end_date:
        date_str = current.strftime("%Y%m%d")
        cache_path = day_cache_dir / f"{date_str}.xml"

        xml_text = _get_text_cached(
            f"{BASE_URL}/DAHourlyOntarioZonalPrice/PUB_DAHourlyOntarioZonalPrice_{date_str}.xml",
            cache_path,
        )
        if xml_text is not None:
            rows = _parse_da_ozp_xml(xml_text, current)
            all_rows.extend(rows)

        current += timedelta(days=1)

    df = pd.DataFrame(all_rows, columns=["datetime", "hour_ending", "da_ozp"])
    df.to_csv(RAW_DIR / "da_ozp_full_history.csv", index=False)
    log.info("DA-OZP history: %d hourly rows across %d days", len(df), n_days)
    return df


def _parse_da_ozp_xml(xml_text: str, report_date: date) -> list[dict]:
    """
    Parses one day's DA-OZP XML into 24 hourly rows.
    NOTE: IESO's XML schema/tag names should be confirmed against a sample file
    the first time you run this -- open one downloaded .xml in a browser or
    text editor and check the tag names below match (they follow the IMRP
    MarketReport schema, but always verify against the real file).
    """
    rows = []
    try:
        root = ET.fromstring(xml_text)
        ns = {"ns": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}

        # Generic search for repeated hourly elements -- adjust the tag name
        # once you've inspected a real sample file
        for hourly_elem in root.iter():
            tag_name = hourly_elem.tag.split("}")[-1]
            if tag_name in ("HourlyPrice", "Hourly", "IMRPHourly"):
                hour = hourly_elem.findtext(".//Hour") or hourly_elem.findtext(".//DeliveryHour")
                price = hourly_elem.findtext(".//EnergyPrice") or hourly_elem.findtext(".//LMP")
                if hour is not None and price is not None:
                    rows.append({
                        "datetime": pd.Timestamp(report_date) + pd.Timedelta(hours=int(hour) - 1),
                        "hour_ending": int(hour),
                        "da_ozp": float(price),
                    })
    except ET.ParseError as e:
        log.warning("Failed to parse XML for %s: %s", report_date, e)

    return rows


# ---------------------------------------------------------------------------
# 4. LFDA (May 2025-present) -- bulk file, NOT per-day
# ---------------------------------------------------------------------------

def fetch_lfda_history() -> pd.DataFrame:
    """
    Downloads the Load Forecast Deviation Adjustment data. Tries the
    'life to date' bulk file first (single file since MRP launch); falls
    back to the year-to-date file if the LTD endpoint doesn't exist.
    """
    ltd_url = f"{BASE_URL}/HourlyLFDALTD/PUB_HourlyLFDALTD.csv"
    df = _get_csv_cached(ltd_url, RAW_DIR / "lfda_life_to_date.csv", skiprows=3)

    if df is None:
        log.info("LTD file not found, falling back to year-to-date file")
        ytd_url = f"{BASE_URL}/HourlyLFDA/PUB_HourlyLFDA.csv"
        df = _get_csv_cached(ytd_url, RAW_DIR / "lfda_ytd.csv", skiprows=3)

    if df is not None:
        df.to_csv(RAW_DIR / "lfda_full_history.csv", index=False)
        log.info("LFDA history: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 5. Combine everything into one continuous price series
# ---------------------------------------------------------------------------

def build_combined_price_series() -> pd.DataFrame:
    """
    Splices legacy HOEP (pre-MRP) with DA-OZP + LFDA (post-MRP) into one
    continuous hourly price series, with a post_mrp flag so the regime
    change is an explicit, visible feature rather than a hidden discontinuity.
    """
    hoep = fetch_hoep_history()
    da_ozp = fetch_da_ozp_history()
    lfda = fetch_lfda_history()

    # --- Pre-MRP segment ---
    hoep_series = hoep.rename(columns={"HOEP": "price"})[["Date", "Hour", "price"]].copy()
    hoep_series["datetime"] = pd.to_datetime(hoep_series["Date"]) + pd.to_timedelta(
        hoep_series["Hour"].astype(int) - 1, unit="h"
    )
    hoep_series["post_mrp"] = False
    hoep_series = hoep_series[["datetime", "price", "post_mrp"]]

    # --- Post-MRP segment: Ontario Price = DA-OZP + LFDA ---
    lfda_series = lfda.rename(columns={lfda.columns[0]: "datetime", "LFDA": "lfda"})
    lfda_series["datetime"] = pd.to_datetime(lfda_series["datetime"])

    merged = da_ozp.merge(lfda_series[["datetime", "lfda"]], on="datetime", how="left")
    merged["price"] = merged["da_ozp"] + merged["lfda"].fillna(0)
    merged["post_mrp"] = True
    ontario_price_series = merged[["datetime", "price", "post_mrp"]]

    combined = pd.concat([hoep_series, ontario_price_series], ignore_index=True)
    combined = combined.sort_values("datetime").drop_duplicates(subset="datetime")

    out_path = RAW_DIR / "combined_price_series.csv"
    combined.to_csv(out_path, index=False)
    log.info("Combined price series: %d rows, saved to %s", len(combined), out_path)
    return combined


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_csv_cached(url: str, cache_path: Path, skiprows: int = 0) -> pd.DataFrame | None:
    """Downloads a CSV (with local caching) and returns it as a DataFrame, or None on failure."""
    if cache_path.exists():
        return pd.read_csv(cache_path, skiprows=skiprows)

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


def _get_text_cached(url: str, cache_path: Path) -> str | None:
    """Downloads raw text (e.g. XML) with local caching, or returns None on failure."""
    if cache_path.exists():
        return cache_path.read_text()

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        cache_path.write_text(resp.text)
        return resp.text
    except requests.exceptions.RequestException as e:
        log.warning("Could not fetch %s: %s", url, e)
        return None
    finally:
        time.sleep(REQUEST_DELAY_SECONDS)


if __name__ == "__main__":
    build_combined_price_series()
