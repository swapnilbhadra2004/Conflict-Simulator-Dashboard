# build_training_panel.py
# Convert training_owid_wb.csv -> training_panel.csv (features + target)

import io, requests
import pandas as pd
import numpy as np
from pathlib import Path

OWID = "https://ourworldindata.org/grapher"
IN_CSV  = "training_owid_wb.csv"
OUT_CSV = "training_panel.csv"

def fetch_brent_prices():
    """Annual Brent price from OWID crude-oil-prices.csv (fallback: empty)."""
    try:
        url = f"{OWID}/crude-oil-prices.csv"
        r = requests.get(url, timeout=40); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df = df.rename(columns={"Entity":"entity","Code":"iso","Year":"year"})
        brent_col = next((c for c in df.columns if "brent" in c.lower() and "price" in c.lower()), None)
        if not brent_col:
            num_cols = [c for c in df.columns if c not in ("entity","iso","year")]
            brent_col = num_cols[0] if num_cols else None
        if not brent_col:
            return pd.DataFrame({"year":[], "brent_price_usd":[]})
        annual = (df.groupby("year")[brent_col].mean()
                    .reset_index().rename(columns={brent_col:"brent_price_usd"}))
        annual["year"] = annual["year"].astype(int)
        return annual
    except Exception:
        return pd.DataFrame({"year":[], "brent_price_usd":[]})

def main():
    if not Path(IN_CSV).exists():
        raise FileNotFoundError(f"{IN_CSV} not found. Run build_training_csv_quick.py first.")

    df = pd.read_csv(IN_CSV)

    for col in ["production","consumption","gdp_usd_billion","gdp_real_growth_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Feature: import dependency (kb/d proxy)
    df["import_dependency_kbpd"] = (df["consumption"].fillna(0) - df["production"].fillna(0)).clip(lower=0)

    # Brent price + YoY %
    brent = fetch_brent_prices()  # year, brent_price_usd
    panel = df.merge(brent, on="year", how="left")
    panel["brent_price_usd"] = panel["brent_price_usd"].ffill().bfill()
    panel = panel.sort_values(["iso","year"])
    panel["brent_change_pct"] = panel.groupby("iso")["brent_price_usd"].pct_change().mul(100).fillna(0.0)

    # Target: GDP growth surprise = real growth - prior 3y rolling avg
    if "gdp_real_growth_pct" not in panel.columns:
        raise ValueError("training_owid_wb.csv missing 'gdp_real_growth_pct' (NY.GDP.MKTP.KD.ZG).")

    roll = (panel.groupby("iso")["gdp_real_growth_pct"]
                 .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean()))
    panel["gdp_growth_roll3"] = roll.values
    panel["gdp_growth_surprise"] = panel["gdp_real_growth_pct"] - panel["gdp_growth_roll3"]

    # Optional placeholders (trainer will include if present)
    for c in ["energy_intensity","fx_reserves_import_months","fuel_subsidy_dummy",
              "manufacturing_share","floating_fx_dummy"]:
        if c not in panel.columns:
            panel[c] = np.nan

    cols = ["iso","country","year","gdp_usd_billion",
            "import_dependency_kbpd","brent_price_usd","brent_change_pct",
            "gdp_real_growth_pct","gdp_growth_roll3","gdp_growth_surprise",
            "energy_intensity","fx_reserves_import_months","fuel_subsidy_dummy",
            "manufacturing_share","floating_fx_dummy"]
    cols = [c for c in cols if c in panel.columns]
    panel = panel[cols].dropna(subset=["gdp_growth_surprise"]).reset_index(drop=True)

    panel.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} with {len(panel):,} rows. Columns: {list(panel.columns)}")

if __name__ == "_main_":
    main()