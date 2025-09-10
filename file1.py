import io
import requests
import pandas as pd
import streamlit as st
import pycountry
import plotly.express as px


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
def iso_to_name(iso):
    """Convert ISO3 code to full country name. Fallback to ISO string if unknown."""
    try:
        country = pycountry.countries.get(alpha_3=iso)
        return country.name if country else iso
    except Exception:
        return iso

def fetch_owid_series(grapher_slug: str) -> pd.DataFrame:
    """Fetch OWID dataset and standardize to: country, iso, year, value"""
    url = f"https://ourworldindata.org/grapher/{grapher_slug}.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))

    
    df = df.rename(columns={"Entity": "country", "Code": "iso", "Year": "year"})
    value_col = [c for c in df.columns if c not in ["country", "iso", "year"]][-1]
    df = df.rename(columns={value_col: "value"})
    return df[["country", "iso", "year", "value"]]

def extract_owid_for_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Filter OWID dataframe to a given year"""
    return df[df["year"] == year].copy()

# Dummy GDP fetcher (replace with API if you want)
def fetch_worldbank_gdp(isos, year_from=2020, year_to=2022) -> pd.DataFrame:
    """Fake GDP fetcher for demo — replace with real API if needed"""
    return pd.DataFrame({
        "iso3": list(isos),
        "year": [year_to] * len(isos),
        "gdp_usd_billion": [500] * len(isos)  
    })

# Dummy Comtrade fetcher
def fetch_comtrade_exports(commodity_code="2709", year=2022) -> pd.DataFrame:
    """Try Comtrade API, but return empty if fails (safe)"""
    url = f"https://comtrade.un.org/api/get?max=50000&type=C&freq=A&px=HS&ps={year}&r=all&p=0&rg=2&cc={commodity_code}&fmt=csv"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        return df[["Reporter ISO", "Trade Value (US$)"]].rename(
            columns={"Reporter ISO": "iso", "Trade Value (US$)": "export_value_usd"}
        )
    except Exception:
        return pd.DataFrame(columns=["iso", "export_value_usd"])

# ---------------------------
# Header / Hero
# ---------------------------
left, right = st.columns([0.7, 0.3])
with left:
    st.markdown('<div class="app-title"><h1>🛢 Conflict Simulator Dashboard</h1></div>', unsafe_allow_html=True)
    st.caption("Real-time what-if economics on oil supply shocks. Clean UI, same logic.")
with right:
    st.markdown(
        '<div style="text-align:right;margin-top:4px">'
        '<span class="pill">OWID</span><span class="pill">World Bank (dummy)</span>'
        '<span class="pill">UN Comtrade (best-effort)</span>'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("⚙ Controls")
commodity_choice = st.sidebar.text_input("Commodity code (HS)", "2709")
year_choice = st.sidebar.number_input("Year", value=2022, min_value=1990, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Model parameters")
severity = st.sidebar.slider("Export cut severity (%)", 0, 100, 50)
price_sensitivity = st.sidebar.slider("Price sensitivity factor", 0.5, 3.0, 1.5, step=0.1)
impact_multiplier = st.sidebar.slider("GDP impact multiplier", 0.1, 1.0, 0.6, step=0.05)

baseline_price = 70.0

# ---------------------------
# Load Data (same sources, nicer messages)
# ---------------------------
with st.spinner("Loading OWID oil production & consumption…"):
    prod_df = fetch_owid_series("oil-production-by-country")
    cons_df = fetch_owid_series("oil-consumption-by-country")

prod_year_df = extract_owid_for_year(prod_df, year_choice).dropna(subset=["iso"])
cons_year_df = extract_owid_for_year(cons_df, year_choice).dropna(subset=["iso"])
isos = set(prod_year_df["iso"].unique()) | set(cons_year_df["iso"].unique())

st.success(f"OWID loaded for {year_choice}: {len(prod_year_df)} production rows · {len(cons_year_df)} consumption rows")

with st.spinner("Fetching trade (exports) data from UN Comtrade…"):
    comtrade_agg = fetch_comtrade_exports(commodity_code=commodity_choice, year=year_choice)
if comtrade_agg.empty:
    st.warning("Comtrade API fetch failed or returned no data. Falling back to OWID net exports only.")

with st.spinner("Fetching GDP data…"):
    wb_gdp = fetch_worldbank_gdp(isos, year_from=year_choice-2, year_to=year_choice)
if wb_gdp.empty or "gdp_usd_billion" not in wb_gdp.columns:
    st.warning("World Bank GDP data not available. Using fallback GDP = 0.")
    wb_gdp = pd.DataFrame(columns=["iso3", "gdp_usd_billion"])
else:
    wb_gdp = wb_gdp.rename(columns={"iso3": "iso"})

# ---------------------------
# Merge master dataset (same logic)
# ---------------------------
master = pd.DataFrame(sorted(list(isos)), columns=["iso"])
master["country"] = master["iso"].apply(iso_to_name)

master = master.merge(prod_year_df[["iso", "value"]].rename(columns={"value": "production"}), on="iso", how="left")
master = master.merge(cons_year_df[["iso", "value"]].rename(columns={"value": "consumption"}), on="iso", how="left")

if not comtrade_agg.empty:
    master = master.merge(comtrade_agg, on="iso", how="left")
else:
    master["export_value_usd"] = 0.0

if not wb_gdp.empty:
    master = master.merge(wb_gdp[["iso", "gdp_usd_billion"]], on="iso", how="left")
else:
    master["gdp_usd_billion"] = 0.0

# Compute exports = production - consumption (fallback)
master["net_exports"] = (master["production"].fillna(0) - master["consumption"].fillna(0)).clip(lower=0)

# ---------------------------
# Scenario Target
# ---------------------------
target_country = st.selectbox("🎯 Choose a country to sanction", master["country"].dropna().unique())
target_iso = master.loc[master["country"] == target_country, "iso"].values[0]
target_exports = master.loc[master["iso"] == target_iso, "net_exports"].values[0]

# ---------------------------
# Simulate (same math, prettier presentation)
# ---------------------------
lost_supply = target_exports * (severity / 100)
total_supply = master["production"].fillna(0).sum()
fraction_loss = lost_supply / total_supply if total_supply > 0 else 0.0

price_increase_fraction = price_sensitivity * fraction_loss
new_price = baseline_price * (1 + price_increase_fraction)

master["import_dependency"] = (master["consumption"].fillna(0) - master["production"].fillna(0)).clip(lower=0)
extra_cost = master["import_dependency"] * (new_price - baseline_price)
master["gdp_impact"] = -(extra_cost / (master["gdp_usd_billion"] * 1000)).fillna(0) * impact_multiplier
master["gdp_change"] = (master["gdp_impact"] * master["gdp_usd_billion"]).fillna(0)

# ---------------------------
# KPIs (metric cards)
# ---------------------------
k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Target exporter</div>
  <div class="metric-value">{target_country}</div>
  <div class="metric-delta subtle">{target_iso}</div>
</div>""", unsafe_allow_html=True)

k2.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Lost supply (relative)</div>
  <div class="metric-value">{severity}%</div>
  <div class="metric-delta subtle">of {target_country}'s net exports</div>
</div>""", unsafe_allow_html=True)

k3.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Baseline oil price</div>
  <div class="metric-value">${baseline_price:,.2f}</div>
</div>""", unsafe_allow_html=True)

k4.markdown(f"""
<div class="metric-card">
  <div class="metric-label">New oil price</div>
  <div class="metric-value">${new_price:,.2f}</div>
  <div class="metric-delta">+{price_increase_fraction*100:.2f}%</div>
</div>""", unsafe_allow_html=True)

st.markdown("")

# ---------------------------
# Tabs: Overview | Impacts | Data
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "💥 Impacts", "📁 Data"])

with tab1:
    st.markdown("### Top GDP Impacts")
    top = master.sort_values("gdp_change", ascending=True).head(15)  # most negative first
    fig = px.bar(
        top,
        x="gdp_change",
        y="country",
        orientation="h",
        title="Projected GDP change (USD billions)",
        labels={"gdp_change": "Δ GDP (USD bn)", "country": ""},
        text=top["gdp_change"].round(2),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### World Map — GDP Change")
    # Plotly expects ISO-3 codes in 'locations'
    map_df = master.copy()
    map_df["iso3"] = map_df["iso"]
    fig_map = px.choropleth(
        map_df,
        locations="iso3",
        color="gdp_change",
        color_continuous_scale="RdBu",
        title="GDP change (USD billions) — darker red = larger loss",
        labels={"gdp_change": "Δ GDP (USD bn)"},
    )
    fig_map.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    c1, c2 = st.columns([0.6, 0.4])

    with c1:
        st.markdown("#### Import Dependency vs. GDP Change")
        bubble = master.copy()
        bubble["size"] = (bubble["import_dependency"].fillna(0) + 1)  # visual size
        fig2 = px.scatter(
            bubble,
            x="import_dependency",
            y="gdp_change",
            size="size",
            hover_name="country",
            labels={"import_dependency": "Import dependency (kb/d proxy)", "gdp_change": "Δ GDP (USD bn)"},
            title="Countries with high import dependency tend to lose more",
        )
        fig2.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown("#### Quick Table (Top 20 loss)")
        tbl = master.sort_values("gdp_change").head(20)[
            ["country", "production", "consumption", "net_exports", "gdp_usd_billion", "gdp_change"]
        ].rename(columns={
            "country": "Country",
            "production": "Production",
            "consumption": "Consumption",
            "net_exports": "Net Exports",
            "gdp_usd_billion": "GDP (USD bn)",
            "gdp_change": "Δ GDP (USD bn)",
        })
        st.dataframe(
            tbl.style.format({
                "Production": "{:,.0f}",
                "Consumption": "{:,.0f}",
                "Net Exports": "{:,.0f}",
                "GDP (USD bn)": "{:,.1f}",
                "Δ GDP (USD bn)": "{:+.2f}",
            }),
            height=480
        )

with tab3:
    st.markdown("#### Merged dataset (sample of 10)")
    st.dataframe(master.head(10))

    # Download button
    csv_bytes = master.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download full results (CSV)", csv_bytes, file_name="conflict_sim_results.csv", mime="text/csv")