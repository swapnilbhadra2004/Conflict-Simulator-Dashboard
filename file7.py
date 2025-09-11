# train_ai_model.py
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_CSV  = sys.argv[1] if len(sys.argv) > 1 else "training_panel.csv"
MODEL_OUT = sys.argv[2] if len(sys.argv) > 2 else "model.joblib"

BASE_FEATS = ["brent_change_pct", "import_dependency_kbpd", "gdp_usd_billion"]
OPTIONAL   = [
    "energy_intensity",
    "fx_reserves_import_months",
    "fuel_subsidy_dummy",
    "manufacturing_share",
    "floating_fx_dummy",
]

def main():
    df = pd.read_csv(DATA_CSV)
    if "gdp_growth_surprise" not in df.columns:
        raise ValueError("training_panel.csv must include 'gdp_growth_surprise'.")

    # Pick features that actually have data (at least 5% non-null)
    cand = BASE_FEATS + OPTIONAL
    present = [c for c in cand if c in df.columns]
    coverage = (df[present].notna().mean() * 100).round(1)  # % non-null
    keep = [c for c in present if coverage[c] >= 5.0]        # drop all-NaN cols

    print("Feature coverage (% non-null):")
    print(coverage.sort_values(ascending=False).to_string())
    print("\nUsing features:", keep, "\n")

    # Filter rows with a target, sort by time
    df = df.dropna(subset=["gdp_growth_surprise"]).sort_values(["year", "iso"]).reset_index(drop=True)
    X = df[keep].copy()
    y = df["gdp_growth_surprise"].astype(float).copy()

    # Preprocess numeric features
    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
        ]
    )
    pre = ColumnTransformer([("num", num_pipe, keep)], remainder="drop")

    # Model
    model = HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.05, max_bins=255, random_state=42
    )
    pipe = Pipeline([("pre", pre), ("model", model)])

    # Time-based CV by year
    years = np.sort(df["year"].unique())
    splits = min(5, max(2, len(years) - 1))
    tscv = TimeSeriesSplit(n_splits=splits)

    preds = np.zeros(len(y))
    for i, (tr, te) in enumerate(tscv.split(years)):
        train_years, test_years = years[tr], years[te]
        mtr = df["year"].isin(train_years)
        mte = df["year"].isin(test_years)
        pipe.fit(X[mtr], y[mtr])
        ph = pipe.predict(X[mte])
        preds[mte] = ph
        print(f"Fold {i+1}: MAE={mean_absolute_error(y[mte], ph):.3f}, R2={r2_score(y[mte], ph):.3f}")

    print("CV overall: MAE={:.3f}, R2={:.3f}".format(
        mean_absolute_error(y, preds), r2_score(y, preds)
    ))

    # Fit on all and save
    pipe.fit(X, y)
    joblib.dump({"pipeline": pipe, "features": keep}, MODEL_OUT)
    print(f"Saved {MODEL_OUT}. Features used: {keep}")

if _name_ == "_main_":
    main()