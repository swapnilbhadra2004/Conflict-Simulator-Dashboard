# build_training_csv_quick.py
# Build a training CSV by merging OWID (oil prod/cons) with World Bank GDP data.
# Fixed merge logic: uses suffixes and then consolidates the country column.

import io
import sys
import time
import json
import requests
import pandas as pd

OWID = "https://ourworldindata.org/grapher"
WB_API = "https://api.worldbank.org/v2"

# ---------- Helpers ----------

def fetch_owid_series(slug: str) -> pd.DataFrame:
    """
    Fetch OWID grapher CSV and return columns: country, iso, year, value
    """
    url = f"{OWID}/{slug}.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # OWID sometimes uses Entity/Code/Year and sometimes country/iso_code/year
    rename_map = {
        "Entity": "country", "Code": "iso", "Year": "year",
        "country": "country", "iso_code": "iso", "year": "year"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    # Find the data column
    value_col = [c for c in df.columns if c not in ["country", "iso", "year", "Entity", "Code", "Year"]][-1]
    df = df.rename(columns={value_col: "value"})
    return df[["country", "iso", "year", "value"]]


def fetch_worldbank_indicator(iso_list, indicator: str, year_from: int, year_to: int) -> pd.DataFrame:
    """
    Fetch a World Bank indicator for multiple countries/years.
    Returns: iso, year, value
    """
    if not iso_list:
        return pd.DataFrame(columns=["iso", "year", "value"])

    # WB API allows semicolon-separated list, but keep request under URL limits
    chunks = []
    iso_list = list(iso_list)
    step = 40
    for i in range(0, len(iso_list), step):
        batch = iso_list[i:i+step]
        url = (f"{WB_API}/country/{';'.join(batch)}/indicator/{indicator}"
               f"?date={year_from}:{year_to}&format=json&per_page=20000")
        r = requests.get(url, timeout=40)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
            chunks.extend(data[1])
        time.sleep(0.2)  # be gentle

    if not chunks:
        return pd.DataFrame(columns=["iso", "year", "value"])

    rows = []
    for item in chunks:
        iso = item.get("countryiso3code")
        year = item.get("date")
        val = item.get("value")
        if iso and year:
            rows.append({"iso": iso, "year": int(year), "value": val})
    df = pd.DataFrame(rows)
    return df


def coalesce_country(country_prod, country_cons):
    """Prefer production country name, fall back to consumption."""
    if pd.notna(country_prod) and str(country_prod).strip():
        return country_prod
    return country_cons


# ---------- Build dataset ----------

def main(year_from=1990, year_to=2024, out_path="training_owid_wb.csv"):
    print("Fetching OWID prod/cons...")
    prod = fetch_owid_series("oil-production-by-country").rename(columns={"value": "production"})
    cons = fetch_owid_series("oil-consumption-by-country").rename(columns={"value": "consumption"})

    # Some OWID rows have aggregates without ISO; drop those for merges
    prod = prod.dropna(subset=["iso"])
    cons = cons.dropna(subset=["iso"])

    isos = sorted(set(prod["iso"].unique()) | set(cons["iso"].unique()))

    print("Fetching WB GDP current USD...")
    gdp_usd = fetch_worldbank_indicator(isos, "NY.GDP.MKTP.CD", year_from, year_to).rename(
        columns={"value": "gdp_current_usd"}
    )

    print("Fetching WB real GDP growth (%)...")
    gdp_growth = fetch_worldbank_indicator(isos, "NY.GDP.MKTP.KD.ZG", year_from, year_to).rename(
        columns={"value": "gdp_real_growth_pct"}
    )

    # Merge prod & cons — use suffixes to avoid the 'country' overlap error
    merged = prod.merge(
        cons,
        on=["iso", "year"],
        how="outer",
        suffixes=("_prod", "_cons")
    )

    # Consolidate country column
    merged["country"] = merged.apply(
        lambda r: coalesce_country(r.get("country_prod"), r.get("country_cons")),
        axis=1
    )
    merged = merged.drop(columns=[c for c in ["country_prod", "country_cons"] if c in merged.columns])

    # Merge GDP (current USD) and growth %
    merged = merged.merge(gdp_usd, on=["iso", "year"], how="left")
    merged = merged.merge(gdp_growth, on=["iso", "year"], how="left")

    # Clean types
    for col in ["production", "consumption", "gdp_current_usd", "gdp_real_growth_pct"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Convenience columns
    merged["gdp_usd_billion"] = merged["gdp_current_usd"] / 1e9
    merged["net_exports_proxy"] = (merged["production"].fillna(0) - merged["consumption"].fillna(0))

    # Order columns
    cols = ["iso", "country", "year",
            "production", "consumption", "net_exports_proxy",
            "gdp_current_usd", "gdp_usd_billion", "gdp_real_growth_pct"]
    cols = [c for c in cols if c in merged.columns]  # keep only existing
    merged = merged[cols].sort_values(["iso", "year"])

    print(f"Writing CSV -> {out_path}")
    merged.to_csv(out_path, index=False)
    print("Done.")

if __name__ == "_main_":
    # You can pass custom args like:
    # py build_training_csv_quick.py 2000 2024 custom.csv
    yf = int(sys.argv[1]) if len(sys.argv) > 1 else 1990
    yt = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    out = sys.argv[3] if len(sys.argv) > 3 else "training_owid_wb.csv"
    main(yf, yt, out)