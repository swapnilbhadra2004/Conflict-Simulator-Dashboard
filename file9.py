# conflict_simulator_best.py
# Deterministic, unit-consistent GDP impact model (no ML)
# Deps: streamlit pandas numpy requests pycountry plotly

import io, os
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pycountry
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(page_title="Conflict Simulator — Deterministic", page_icon="🛢", layout="wide")

def _session(retries=1, backoff=0.4, timeout=20):
    s = requests.Session()
    r = Retry(total=retries, connect=retries, read=retries, status=retries,
              backoff_factor=backoff, status_forcelist=[429,500,502,503,504],
              allowed_methods=["GET"], raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    s.timeout = timeout
    return s

HTTP = _session()

def iso_to_name(iso):
    try:
        c = pycountry.countries.get(alpha_3=iso)
        return c.name if c else iso
    except Exception:
        return iso

def _normalize_owid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"Entity": "country", "Code": "iso", "Year": "year"})
    value_col = [c for c in df.columns if c not in ("country", "iso", "year")][-1]
    df = df.rename(columns={value_col: "value"})
    return df[["country", "iso", "year", "value"]]

@st.cache_data(show_spinner=False)
def fetch_owid_series(slug: str) -> pd.DataFrame:
    url = f"https://ourworldindata.org/grapher/{slug}.csv"
    resp = HTTP.get(url)
    resp.raise_for_status()
    return _normalize_owid(pd.read_csv(io.StringIO(resp.text)))

def extract_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[df["year"] == year].copy()

@st.cache_data(show_spinner=False)
def fetch_worldbank_gdp_or_fallback(isos, year: int) -> pd.DataFrame:
    try:
        iso_list = list(isos)[:50]
        url = f"http://api.worldbank.org/v2/country/{';'.join(iso_list)}/indicator/NY.GDP.MKTP.CD?format=json&date={year}"
        r = HTTP.get(url)
        r.raise_for_status()
        data = r.json()
        rows = []
        if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
            for item in data[1]:
                if item.get("value") is not None:
                    rows.append({"iso": item["country"]["id"], "gdp_usd_billion": float(item["value"]) / 1e9})
        if rows:
            return pd.DataFrame(rows)
    except Exception:
        pass
    fallback = {'USA':23315,'CHN':17734,'JPN':4937,'DEU':4072,'IND':3385,'GBR':3070,'FRA':2782,'ITA':2059,
                'CAN':2015,'KOR':1798,'RUS':1776,'BRA':1609,'AUS':1542,'ESP':1387,'MEX':1274,'IDN':1186}
    return pd.DataFrame({"iso":[i for i in isos],
                         "gdp_usd_billion":[fallback.get(i[-3:],50.0) for i in isos]})

left, right = st.columns([0.7, 0.3])
with left:
    st.markdown("## 🛢 Conflict Simulator — Deterministic")
    st.caption("Transparent, unit-consistent GDP impact estimate from oil supply shocks (no ML).")
with right:
    st.markdown('<div style="text-align:right"><span style="padding:4px 10px;border:1px solid #8882;border-radius:999px;margin-left:6px;">OWID</span><span style="padding:4px 10px;border:1px solid #8882;border-radius:999px;margin-left:6px;">World Bank / fallback</span></div>', unsafe_allow_html=True)

st.sidebar.header("⚙ Controls")
year_choice = st.sidebar.number_input("Year", value=2022, min_value=1990, step=1)
severity = st.sidebar.slider("Export cut severity (target exporter)", 0, 100, 50)
price_sensitivity = st.sidebar.slider("Price sensitivity (global)", 0.5, 3.0, 1.5, step=0.1)
impact_multiplier = st.sidebar.slider("GDP impact multiplier (2nd-round effects)", 0.1, 1.5, 0.6, step=0.05)

st.sidebar.markdown("---")
oil_pass_through = st.sidebar.slider("Oil price pass-through to import bill", 0.2, 1.0, 0.7, step=0.05)
import_substitution = st.sidebar.slider("Short-run substitution (cut in oil usage)", 0.0, 0.5, 0.15, step=0.05)

st.sidebar.markdown("---")
baseline_price = st.sidebar.number_input("Baseline oil price (USD/bbl)", value=70.0, step=1.0)

with st.spinner("Loading OWID oil production & consumption…"):
    prod_df = fetch_owid_series("oil-production-by-country")
    cons_df = fetch_owid_series("oil-consumption-by-country")

prod_year = extract_year(prod_df, year_choice).dropna(subset=["iso"])
cons_year = extract_year(cons_df, year_choice).dropna(subset=["iso"])
isos = set(prod_year["iso"]) | set(cons_year["iso"])

with st.spinner("Fetching GDP…"):
    gdp_df = fetch_worldbank_gdp_or_fallback(isos, year_choice)

master = pd.DataFrame(sorted(list(isos)), columns=["iso"])
master["country"] = master["iso"].apply(iso_to_name)
master = master.merge(prod_year[["iso", "value"]].rename(columns={"value": "production_kbpd"}), on="iso", how="left")
master = master.merge(cons_year[["iso", "value"]].rename(columns={"value": "consumption_kbpd"}), on="iso", how="left")
master = master.merge(gdp_df[["iso", "gdp_usd_billion"]], on="iso", how="left")

master[["production_kbpd", "consumption_kbpd", "gdp_usd_billion"]] = master[["production_kbpd", "consumption_kbpd", "gdp_usd_billion"]].fillna(0.0)
master["net_exports_kbpd"] = (master["production_kbpd"] - master["consumption_kbpd"]).clip(lower=0)
master["import_dependency_kbpd"] = (master["consumption_kbpd"] - master["production_kbpd"]).clip(lower=0)

target_country = st.selectbox("🎯 Target exporter", master.loc[master["net_exports_kbpd"] > 0, "country"].sort_values())
target_iso = master.loc[master["country"] == target_country, "iso"].values[0]
target_exports_kbpd = master.loc[master["iso"] == target_iso, "net_exports_kbpd"].values[0]

lost_supply_kbpd = target_exports_kbpd * (severity / 100.0)
total_production_kbpd = master["production_kbpd"].sum()
fractional_loss = (lost_supply_kbpd / total_production_kbpd) if total_production_kbpd > 0 else 0.0

price_increase_fraction = price_sensitivity * fractional_loss
new_price = baseline_price * (1.0 + price_increase_fraction)
delta_price = new_price - baseline_price

kbpd_to_bpd = 1000.0
annualized_extra_barrels = master["import_dependency_kbpd"] * kbpd_to_bpd * 365.0
extra_spend_usd = annualized_extra_barrels * delta_price
net_extra_spend_usd = extra_spend_usd * oil_pass_through * (1.0 - import_substitution)

gdp_usd = (master["gdp_usd_billion"] * 1e9).replace(0, np.nan)
gdp_impact_pct = -(net_extra_spend_usd / gdp_usd) * 100.0
gdp_impact_pct *= impact_multiplier
gdp_impact_pct = gdp_impact_pct.clip(lower=-25.0, upper=5.0).fillna(0.0)

master["gdp_impact_percent"] = gdp_impact_pct
master["gdp_change_bn"] = (master["gdp_impact_percent"] / 100.0) * master["gdp_usd_billion"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Target exporter", target_country, target_iso)
c2.metric("Lost supply (kb/d)", f"{lost_supply_kbpd:,.0f}")
c3.metric("Baseline price", f"${baseline_price:,.2f}")
c4.metric("New price", f"${new_price:,.2f}", f"+{price_increase_fraction*100:.2f}%")

tab1, tab2 = st.tabs(["📊 Overview", "📁 Data"])

with tab1:
    st.markdown("### Top GDP losses (USD billions)")
    top = master.sort_values("gdp_change_bn").head(15)
    fig = px.bar(top, x="gdp_change_bn", y="country", orientation="h",
                 labels={"gdp_change_bn": "Δ GDP (USD bn)", "country": ""},
                 text=top["gdp_change_bn"].round(2),
                 title="Deterministic estimate of GDP change")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### World map — GDP impact (% of GDP)")
    map_df = master.copy(); map_df["iso3"] = map_df["iso"]
    fig_map = px.choropleth(
        map_df,
        locations="iso3",
        color="gdp_impact_percent",
        color_continuous_scale="RdBu",
        range_color=[-5, 0],
        labels={"gdp_impact_percent": "Δ GDP (%)"},
        title="Deterministic impact (negative = loss)",
        hover_data={"country": True, "gdp_impact_percent": ":.2f", "gdp_change_bn": ":.2f"},
    )
    fig_map.update_geos(showcountries=True, countrycolor="#666", countrywidth=0.7,
                        showcoastlines=True, coastlinecolor="#666",
                        showframe=False, projection_type="natural earth")
    fig_map.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    st.markdown("### Merged dataset (sample of 12)")
    show = master[[
        "country", "production_kbpd", "consumption_kbpd", "net_exports_kbpd",
        "import_dependency_kbpd", "gdp_usd_billion", "gdp_impact_percent", "gdp_change_bn"
    ]].rename(columns={
        "production_kbpd":"Production (kb/d)",
        "consumption_kbpd":"Consumption (kb/d)",
        "net_exports_kbpd":"Net exports (kb/d)",
        "import_dependency_kbpd":"Import dep. (kb/d)",
        "gdp_usd_billion":"GDP (USD bn)",
        "gdp_impact_percent":"Δ GDP (%)",
        "gdp_change_bn":"Δ GDP (USD bn)",
    })
    st.dataframe(
        show.head(12).style.format({
            "Production (kb/d)":"{:,.0f}",
            "Consumption (kb/d)":"{:,.0f}",
            "Net exports (kb/d)":"{:,.0f}",
            "Import dep. (kb/d)":"{:,.0f}",
            "GDP (USD bn)":"{:,.1f}",
            "Δ GDP (%)":"{:+.2f}",
            "Δ GDP (USD bn)":"{:+.2f}",
        }),
        height=360
    )

st.markdown("""
---
*Method*: Extra oil import bill = (import dependency in kb/d × 1000 × 365) × ΔPrice.  
Impact (% of GDP) = − (extra bill / GDP) × pass-through × (1 − substitution) × multiplier.  
Clamped to [−25%, +5%] for sanity on very small economies.
""")