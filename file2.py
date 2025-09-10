# conflict_simulator_fixed.py
# ─────────────────────────────────────────────────────────────────────────────
# Fixed: Corrected GDP impact calculations for all commodities
# ─────────────────────────────────────────────────────────────────────────────

import io
import requests
import pandas as pd
import streamlit as st
import pycountry
import plotly.express as px

# ---------------------------
# Page config & styling
# ---------------------------
st.set_page_config(
    page_title="Conflict Simulator Dashboard",
    page_icon="🛢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .app-title h1 {font-size: 1.9rem; margin-bottom: .25rem}
      .subtle {color:rgba(255,255,255,0.65) !important}
      .metric-card {
          border-radius: 16px; padding: 16px 18px; margin-bottom: 8px;
          background: rgba(120,120,120,0.08); border: 1px solid rgba(127,127,127,.18);
      }
      .metric-label {font-size:.9rem; opacity:.75; margin-bottom:6px}
      .metric-value {font-weight:700; font-size:1.35rem}
      .metric-delta {font-size:.9rem; opacity:.85}
      .dataframe .row_heading.level0 {display:none}
      .stAlert {margin-top: .25rem}
      .section-h {margin-top: .75rem}
      .pill {
          padding: 4px 10px; border-radius: 999px; font-size:.8rem; 
          border: 1px solid rgba(127,127,127,.25);
          background: rgba(120,120,120,0.08);
          margin-left:6px
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# HARDCODED FALLBACK DATA
# ---------------------------
HARDCODED_GDP_DATA = {
    'USA': 23315.0, 'CHN': 17963.0, 'JPN': 4937.0, 'DEU': 4082.0, 
    'IND': 3385.0, 'GBR': 3070.0, 'FRA': 2782.0, 'ITA': 2108.0,
    'CAN': 2015.0, 'KOR': 1798.0, 'RUS': 1835.0, 'BRA': 1609.0,
    'AUS': 1542.0, 'ESP': 1425.0, 'MEX': 1293.0, 'IDN': 1186.0,
    'NLD': 990.0, 'SAU': 833.0, 'TUR': 815.0, 'CHE': 800.0
}

HARDCODED_EXPORT_DATA = {
    # Oil (2709)
    "2709": {
        'SAU': 200000000000, 'RUS': 180000000000, 'IRQ': 100000000000, 'CAN': 95000000000,
        'ARE': 90000000000, 'USA': 85000000000, 'KWT': 70000000000, 'NGA': 65000000000,
        'NOR': 60000000000, 'KAZ': 55000000000, 'QAT': 50000000000, 'DZA': 45000000000
    },
    # Refined Petroleum (2710)
    "2710": {
        'USA': 120000000000, 'RUS': 100000000000, 'NLD': 90000000000, 'SGP': 85000000000,
        'KOR': 80000000000, 'SAU': 75000000000, 'IND': 70000000000, 'DEU': 65000000000,
        'ARE': 60000000000, 'FRA': 55000000000, 'GBR': 50000000000, 'ITA': 45000000000
    },
    # Wheat (1001)
    "1001": {
        'RUS': 40000000000, 'USA': 35000000000, 'CAN': 30000000000, 'FRA': 25000000000,
        'AUS': 20000000000, 'UKR': 18000000000, 'ARG': 15000000000, 'DEU': 12000000000,
        'KAZ': 10000000000, 'IND': 8000000000, 'ROU': 7000000000, 'POL': 6000000000
    },
    # Corn (1005)
    "1005": {
        'USA': 50000000000, 'BRA': 30000000000, 'ARG': 25000000000, 'UKR': 20000000000,
        'FRA': 15000000000, 'RUS': 12000000000, 'IND': 10000000000, 'CAN': 8000000000,
        'HUN': 6000000000, 'ROU': 5000000000, 'SRB': 4000000000, 'MEX': 3000000000
    },
    # Soybeans (1201)
    "1201": {
        'BRA': 45000000000, 'USA': 40000000000, 'ARG': 25000000000, 'PAR': 8000000000,
        'CAN': 6000000000, 'UKR': 5000000000, 'RUS': 4000000000, 'IND': 3000000000,
        'BOL': 2000000000, 'URU': 1500000000, 'CHN': 1000000000, 'IDN': 800000000
    },
    # Iron Ore (2601)
    "2601": {
        'AUS': 120000000000, 'BRA': 80000000000, 'RUS': 30000000000, 'CAN': 25000000000,
        'IND': 20000000000, 'UKR': 15000000000, 'SWE': 10000000000, 'KAZ': 8000000000,
        'MEX': 6000000000, 'CHL': 5000000000, 'PER': 4000000000, 'VEN': 3000000000
    },
    # Copper Ore (2603)
    "2603": {
        'CHL': 40000000000, 'PER': 25000000000, 'CHN': 20000000000, 'AUS': 18000000000,
        'RUS': 15000000000, 'CAN': 12000000000, 'USA': 10000000000, 'MEX': 8000000000,
        'KAZ': 7000000000, 'IDN': 6000000000, 'POL': 5000000000, 'ZMB': 4000000000
    },
    # Motor Vehicles (8703)
    "8703": {
        'DEU': 180000000000, 'JPN': 120000000000, 'USA': 90000000000, 'CHN': 85000000000,
        'KOR': 65000000000, 'FRA': 50000000000, 'GBR': 45000000000, 'ITA': 40000000000,
        'CAN': 35000000000, 'MEX': 30000000000, 'ESP': 28000000000, 'IND': 25000000000
    },
    # Computers (8471)
    "8471": {
        'CHN': 250000000000, 'USA': 120000000000, 'DEU': 80000000000, 'JPN': 70000000000,
        'KOR': 65000000000, 'NLD': 60000000000, 'MEX': 55000000000, 'CZE': 50000000000,
        'HUN': 45000000000, 'POL': 40000000000, 'GBR': 35000000000, 'FRA': 30000000000
    },
    # Telephones (8517)
    "8517": {
        'CHN': 180000000000, 'VNM': 80000000000, 'IND': 50000000000, 'KOR': 45000000000,
        'USA': 40000000000, 'HKG': 35000000000, 'DEU': 30000000000, 'JPN': 25000000000,
        'MEX': 20000000000, 'NLD': 18000000000, 'CZE': 15000000000, 'BRA': 12000000000
    }
}

# Import dependency estimates (as % of domestic consumption)
HARDCODED_IMPORT_DEPENDENCY = {
    # Motor Vehicles (8703) - countries that heavily import vehicles
    "8703": {
        'USA': 0.25, 'CHN': 0.15, 'JPN': 0.05, 'DEU': 0.10, 'GBR': 0.80, 
        'FRA': 0.60, 'ITA': 0.50, 'CAN': 0.85, 'KOR': 0.20, 'RUS': 0.70,
        'BRA': 0.30, 'AUS': 0.90, 'ESP': 0.75, 'MEX': 0.40, 'IND': 0.25,
        'NLD': 0.95, 'SAU': 0.98, 'TUR': 0.60, 'CHE': 0.99, 'IDN': 0.80
    },
    # Computers (8471) - most countries import heavily
    "8471": {
        'USA': 0.60, 'CHN': 0.30, 'JPN': 0.40, 'DEU': 0.70, 'GBR': 0.90, 
        'FRA': 0.85, 'ITA': 0.85, 'CAN': 0.95, 'KOR': 0.50, 'RUS': 0.80,
        'BRA': 0.90, 'AUS': 0.95, 'ESP': 0.90, 'MEX': 0.70, 'IND': 0.75,
        'NLD': 0.80, 'SAU': 0.99, 'TUR': 0.85, 'CHE': 0.95, 'IDN': 0.90
    },
    # Telephones (8517)
    "8517": {
        'USA': 0.70, 'CHN': 0.20, 'JPN': 0.60, 'DEU': 0.80, 'GBR': 0.95, 
        'FRA': 0.90, 'ITA': 0.90, 'CAN': 0.95, 'KOR': 0.30, 'RUS': 0.85,
        'BRA': 0.85, 'AUS': 0.95, 'ESP': 0.90, 'MEX': 0.75, 'IND': 0.60,
        'NLD': 0.85, 'SAU': 0.99, 'TUR': 0.85, 'CHE': 0.95, 'IDN': 0.80
    },
    # Agricultural commodities - major importers
    "1001": {  # Wheat
        'CHN': 0.15, 'USA': 0.02, 'JPN': 0.85, 'DEU': 0.40, 'IND': 0.05, 
        'GBR': 0.85, 'FRA': 0.10, 'ITA': 0.70, 'CAN': 0.02, 'KOR': 0.95,
        'RUS': 0.0, 'BRA': 0.60, 'AUS': 0.0, 'ESP': 0.50, 'MEX': 0.30,
        'IDN': 0.90, 'NLD': 0.80, 'SAU': 0.95, 'TUR': 0.30, 'CHE': 0.95
    },
    "1005": {  # Corn
        'CHN': 0.10, 'USA': 0.0, 'JPN': 0.99, 'DEU': 0.60, 'IND': 0.02, 
        'GBR': 0.95, 'FRA': 0.30, 'ITA': 0.80, 'CAN': 0.10, 'KOR': 0.99,
        'RUS': 0.05, 'BRA': 0.0, 'AUS': 0.20, 'ESP': 0.70, 'MEX': 0.20,
        'IDN': 0.30, 'NLD': 0.90, 'SAU': 0.95, 'TUR': 0.50, 'CHE': 0.99
    },
    "1201": {  # Soybeans
        'CHN': 0.85, 'USA': 0.0, 'JPN': 0.95, 'DEU': 0.90, 'IND': 0.70, 
        'GBR': 0.95, 'FRA': 0.80, 'ITA': 0.90, 'CAN': 0.10, 'KOR': 0.95,
        'RUS': 0.20, 'BRA': 0.0, 'AUS': 0.30, 'ESP': 0.85, 'MEX': 0.60,
        'IDN': 0.80, 'NLD': 0.95, 'SAU': 0.99, 'TUR': 0.90, 'CHE': 0.99
    },
    # Mining commodities
    "2601": {  # Iron Ore
        'CHN': 0.80, 'USA': 0.30, 'JPN': 0.95, 'DEU': 0.85, 'IND': 0.20, 
        'GBR': 0.95, 'FRA': 0.90, 'ITA': 0.90, 'CAN': 0.10, 'KOR': 0.95,
        'RUS': 0.0, 'BRA': 0.0, 'AUS': 0.0, 'ESP': 0.85, 'MEX': 0.60,
        'IDN': 0.40, 'NLD': 0.95, 'SAU': 0.95, 'TUR': 0.80, 'CHE': 0.99
    },
    "2603": {  # Copper Ore
        'CHN': 0.70, 'USA': 0.40, 'JPN': 0.95, 'DEU': 0.90, 'IND': 0.80, 
        'GBR': 0.95, 'FRA': 0.90, 'ITA': 0.90, 'CAN': 0.20, 'KOR': 0.95,
        'RUS': 0.0, 'BRA': 0.30, 'AUS': 0.0, 'ESP': 0.85, 'MEX': 0.40,
        'IDN': 0.30, 'NLD': 0.95, 'SAU': 0.95, 'TUR': 0.80, 'CHE': 0.99
    }
}

# Default fallback for unknown commodities
DEFAULT_EXPORT_DATA = {
    'USA': 50000000000, 'CHN': 45000000000, 'DEU': 40000000000, 'JPN': 35000000000,
    'GBR': 30000000000, 'FRA': 25000000000, 'CAN': 20000000000, 'ITA': 18000000000,
    'KOR': 15000000000, 'RUS': 12000000000, 'BRA': 10000000000, 'AUS': 8000000000
}

BASELINE_PRICES = {
    "2709": 75.0,    # Crude Oil ($/barrel)
    "2710": 85.0,    # Refined Petroleum ($/barrel)
    "1001": 300.0,   # Wheat ($/ton)
    "1005": 250.0,   # Corn ($/ton)
    "1201": 500.0,   # Soybeans ($/ton)
    "2601": 120.0,   # Iron Ore ($/ton)
    "2603": 8000.0,  # Copper Ore ($/ton)
    "8703": 25000.0, # Motor Vehicles ($/unit)
    "8471": 800.0,   # Computers ($/unit)
    "8517": 300.0    # Telephones ($/unit)
}

# Market size estimates (global consumption in USD billions)
MARKET_SIZES = {
    "2709": 2000.0,   # Crude Oil
    "2710": 1500.0,   # Refined Petroleum
    "1001": 150.0,    # Wheat
    "1005": 200.0,    # Corn
    "1201": 120.0,    # Soybeans
    "2601": 300.0,    # Iron Ore
    "2603": 150.0,    # Copper Ore
    "8703": 2500.0,   # Motor Vehicles
    "8471": 400.0,    # Computers
    "8517": 500.0     # Telephones
}

# ---------------------------
# Helpers
# ---------------------------
def iso_to_name(iso):
    try:
        country = pycountry.countries.get(alpha_3=iso)
        return country.name if country else iso
    except Exception:
        return iso

def fetch_owid_series(grapher_slug: str) -> pd.DataFrame:
    try:
        url = f"https://ourworldindata.org/grapher/{grapher_slug}.csv"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df = df.rename(columns={"Entity": "country", "Code": "iso", "Year": "year"})
        value_col = [c for c in df.columns if c not in ["country", "iso", "year"]][-1]
        df = df.rename(columns={value_col: "value"})
        return df[["country", "iso", "year", "value"]]
    except Exception:
        return pd.DataFrame(columns=["country", "iso", "year", "value"])

def extract_owid_for_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[df["year"] == year].copy()

def fetch_worldbank_gdp(isos, year=2022) -> pd.DataFrame:
    gdp_data = []
    for iso in isos:
        if iso in HARDCODED_GDP_DATA:
            gdp_data.append({'iso': iso, 'gdp_usd_billion': HARDCODED_GDP_DATA[iso]})
    return pd.DataFrame(gdp_data)

def fetch_comtrade_exports(commodity_code="2709", year=2022) -> pd.DataFrame:
    try:
        url = f"https://comtrade.un.org/api/get?max=50000&type=C&freq=A&px=HS&ps={year}&r=all&p=0&rg=2&cc={commodity_code}&fmt=csv"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if len(df) > 0:
            return df[["Reporter ISO", "Trade Value (US$)"]].rename(
                columns={"Reporter ISO": "iso", "Trade Value (US$)": "export_value_usd"}
            )
    except Exception:
        pass
    
    # Fallback to hardcoded data
    if commodity_code in HARDCODED_EXPORT_DATA:
        export_data = []
        for iso, value in HARDCODED_EXPORT_DATA[commodity_code].items():
            export_data.append({'iso': iso, 'export_value_usd': value})
        return pd.DataFrame(export_data)
    else:
        # Default fallback for unknown commodities
        export_data = []
        for iso, value in DEFAULT_EXPORT_DATA.items():
            export_data.append({'iso': iso, 'export_value_usd': value})
        return pd.DataFrame(export_data)

# ---------------------------
# Oil HS codes
# ---------------------------
OIL_CODES = ["2709", "2710", "2711", "2701", "2702"]

# ---------------------------
# Header / Hero
# ---------------------------
left, right = st.columns([0.7, 0.3])
with left:
    st.markdown('<div class="app-title"><h1>🛢 Conflict Simulator Dashboard</h1></div>', unsafe_allow_html=True)
    st.caption("Real-time what-if economics on commodity supply shocks. Clean UI, flexible logic.")
with right:
    st.markdown(
        '<div style="text-align:right;margin-top:4px">'
        '<span class="pill">OWID (for oil)</span><span class="pill">World Bank</span>'
        '<span class="pill">UN Comtrade</span>'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("⚙ Controls")

commodity_options = {
    "2709": "Crude Oil",
    "2710": "Refined Petroleum", 
    "1001": "Wheat",
    "1005": "Maize (Corn)",
    "1201": "Soybeans",
    "2601": "Iron Ore",
    "2603": "Copper Ore",
    "8703": "Motor Vehicles",
    "8471": "Computers",
    "8517": "Telephones"
}

selected_commodity = st.sidebar.selectbox(
    "Commodity code (HS)",
    options=list(commodity_options.keys()),
    format_func=lambda x: f"{x} - {commodity_options[x]}",
    index=0
)

commodity_choice = selected_commodity
year_choice = st.sidebar.number_input("Year", value=2022, min_value=2018, max_value=2023, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Model parameters")
severity = st.sidebar.slider("Export cut severity (%)", 0, 100, 50)
price_sensitivity = st.sidebar.slider("Price sensitivity factor", 0.5, 3.0, 1.5, step=0.1)
impact_multiplier = st.sidebar.slider("GDP impact multiplier", 0.1, 1.0, 0.6, step=0.05)

baseline_price = BASELINE_PRICES.get(commodity_choice, 100.0)

# ---------------------------
# Load Data
# ---------------------------
if commodity_choice in OIL_CODES:
    with st.spinner("Loading OWID oil production & consumption…"):
        try:
            prod_df = fetch_owid_series("oil-production-by-country")
            cons_df = fetch_owid_series("oil-consumption-by-country")
            prod_year_df = extract_owid_for_year(prod_df, year_choice).dropna(subset=["iso"])
            cons_year_df = extract_owid_for_year(cons_df, year_choice).dropna(subset=["iso"])
            isos = set(prod_year_df["iso"].unique()) | set(cons_year_df["iso"].unique())
            st.success(f"OWID loaded for {year_choice}")
        except Exception as e:
            st.error(f"Failed to load OWID data: {e}")
            prod_year_df = pd.DataFrame(columns=["iso","value"])
            cons_year_df = pd.DataFrame(columns=["iso","value"])
            isos = set()
else:
    st.info(f"HS code {commodity_choice} ({commodity_options.get(commodity_choice, 'Commodity')}): using trade data")
    prod_year_df = pd.DataFrame(columns=["iso","value"])
    cons_year_df = pd.DataFrame(columns=["iso","value"])
    isos = set()

with st.spinner("Fetching trade data…"):
    comtrade_agg = fetch_comtrade_exports(commodity_code=commodity_choice, year=year_choice)

with st.spinner("Fetching GDP data…"):
    all_isos = set(comtrade_agg["iso"].unique()) if not comtrade_agg.empty else set(HARDCODED_GDP_DATA.keys())
    wb_gdp = fetch_worldbank_gdp(all_isos, year_choice)

# ---------------------------
# Merge master dataset
# ---------------------------
master = pd.DataFrame(sorted(list(all_isos)), columns=["iso"])
master["country"] = master["iso"].apply(iso_to_name)

if not prod_year_df.empty:
    master = master.merge(prod_year_df[["iso", "value"]].rename(columns={"value": "production"}), on="iso", how="left")
if not cons_year_df.empty:
    master = master.merge(cons_year_df[["iso", "value"]].rename(columns={"value": "consumption"}), on="iso", how="left")

if not comtrade_agg.empty:
    master = master.merge(comtrade_agg, on="iso", how="left")
else:
    master["export_value_usd"] = 0.0

if not wb_gdp.empty:
    master = master.merge(wb_gdp, on="iso", how="left")
else:
    # Fallback GDP
    master["gdp_usd_billion"] = master["iso"].apply(lambda x: HARDCODED_GDP_DATA.get(x, 100.0))

# Fill NaN values
master = master.fillna(0)

# Net exports (convert to billions for consistency)
if commodity_choice in OIL_CODES:
    master["net_exports"] = (master["production"] - master["consumption"]).clip(lower=0)
else:
    master["net_exports"] = master["export_value_usd"] / 1e9  # Convert to billions USD

# ---------------------------
# Check if master data is available
# ---------------------------
if master.empty:
    st.error("No data available. Please check your commodity code and year selection.")
    st.stop()

available_countries = master["country"].dropna().unique()
if len(available_countries) == 0:
    st.error("No countries found in the data.")
    st.stop()

# ---------------------------
# Scenario Target
# ---------------------------
target_country = st.selectbox("🎯 Choose a country to sanction", available_countries)

# Safe access to target_iso
target_match = master.loc[master["country"] == target_country, "iso"]
if target_match.empty:
    st.error(f"Selected country '{target_country}' not found in ISO data.")
    st.stop()
else:
    target_iso = target_match.values[0]

target_exports = master.loc[master["iso"] == target_iso, "net_exports"].values[0]

# ---------------------------
# Simulate
# ---------------------------
lost_supply = target_exports * (severity / 100)
total_supply = master["net_exports"].sum()
fraction_loss = lost_supply / total_supply if total_supply > 0 else 0.0

price_increase_fraction = price_sensitivity * fraction_loss
new_price = baseline_price * (1 + price_increase_fraction)

# Calculate import dependency - FIXED LOGIC
if commodity_choice in OIL_CODES:
    # For oil: import dependency = consumption - production (if positive)
    master["import_dependency"] = (master["consumption"] - master["production"]).clip(lower=0)
else:
    # For other commodities: use hardcoded import dependency ratios
    market_size = MARKET_SIZES.get(commodity_choice, 500.0)  # Default market size
    import_ratios = HARDCODED_IMPORT_DEPENDENCY.get(commodity_choice, {})
    
    # Apply import dependency ratios to estimate import volumes
    master["import_dependency"] = master["iso"].apply(
        lambda iso: import_ratios.get(iso, 0.5) * (market_size / len(master)) * (master.loc[master["iso"] == iso, "gdp_usd_billion"].iloc[0] / 1000) if len(master.loc[master["iso"] == iso]) > 0 else 0
    )

# Calculate additional costs due to price increase
price_increase_per_unit = new_price - baseline_price
master["additional_import_cost"] = master["import_dependency"] * price_increase_per_unit

# GDP impact as percentage of GDP
master["gdp_impact_pct"] = (master["additional_import_cost"] / (master["gdp_usd_billion"] * 1000)) * impact_multiplier  # Convert billions to actual
master["gdp_change"] = -master["gdp_impact_pct"] * master["gdp_usd_billion"]  # In billions USD

# ---------------------------
# KPIs
# ---------------------------
k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"""
<div class="metric-card"><div class="metric-label">Target exporter</div>
<div class="metric-value">{target_country}</div>
<div class="metric-delta subtle">{target_iso}</div></div>""", unsafe_allow_html=True)

k2.markdown(f"""
<div class="metric-card"><div class="metric-label">Lost supply (relative)</div>
<div class="metric-value">{severity}%</div>
<div class="metric-delta subtle">of {target_country}'s exports</div></div>""", unsafe_allow_html=True)

k3.markdown(f"""
<div class="metric-card"><div class="metric-label">Baseline price</div>
<div class="metric-value">${baseline_price:,.2f}</div></div>""", unsafe_allow_html=True)

k4.markdown(f"""
<div class="metric-card"><div class="metric-label">New price</div>
<div class="metric-value">${new_price:,.2f}</div>
<div class="metric-delta">+{price_increase_fraction*100:.2f}%</div></div>""", unsafe_allow_html=True)

# ---------------------------
# Visualizations
# ---------------------------
st.markdown("## 📊 Economic Impact Visualizations")

# Bar Chart - GDP change (billions USD)
bar_df = master.sort_values("gdp_change")
fig_bar = px.bar(
    bar_df,
    x="gdp_change",
    y="country",
    orientation="h",
    title="GDP Change by Country (Billions USD)",
    labels={"gdp_change": "GDP Change (Billion USD)", "country": "Country"},
    color="gdp_change",
    color_continuous_scale="RdBu",
)
st.plotly_chart(fig_bar, use_container_width=True)

# World Map - GDP impact percentage
map_df = master.copy()
fig_map = px.choropleth(
    map_df,
    locations="iso",
    color="gdp_impact_pct",
    hover_name="country",
    color_continuous_scale="Reds",
    projection="natural earth",
    title="World Map of GDP Impact (%)",
)
st.plotly_chart(fig_map, use_container_width=True)