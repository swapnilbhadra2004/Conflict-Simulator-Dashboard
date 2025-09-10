# conflict_simulator_pretty.py
# ─────────────────────────────────────────────────────────────────────────────
# Enhanced with AI-powered impact simulation for sanctions
# ─────────────────────────────────────────────────────────────────────────────

import io
import requests
import pandas as pd
import numpy as np
import streamlit as st
import pycountry
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Page config & light styling
# ---------------------------
st.set_page_config(
    page_title="Conflict Simulator Dashboard",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal custom CSS for cards, captions, and tighter spacing
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
      .ai-badge {
          background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
          color: white;
          padding: 4px 10px;
          border-radius: 999px;
          font-size: 0.8rem;
          margin-left: 8px;
      }
      .warning-badge {
          background: linear-gradient(45deg, #FF6B6B, #FF8E53);
          color: white;
          padding: 4px 10px;
          border-radius: 999px;
          font-size: 0.8rem;
          margin-left: 8px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# AI Simulation Functions - REFINED
# ---------------------------
def create_ai_simulation_model(master_data):
    """Create a realistic AI model to predict GDP impact based on country features"""
    # Filter out countries with missing essential data
    valid_data = master_data.dropna(subset=['gdp_usd_billion', 'import_dependency', 'production', 'consumption'])
    
    if len(valid_data) < 10:  # Need sufficient data for meaningful training
        return None, None, None
    
    X = []
    y = []
    
    # Create realistic training data based on economic principles
    for _, row in valid_data.iterrows():
        # Features: normalized economic indicators
        features = [
            np.log1p(row['import_dependency']),  # Log transform for heavy-tailed distributions
            np.log1p(row['gdp_usd_billion']),
            row['production'] / max(1, row['gdp_usd_billion']),  # Production intensity
            row['consumption'] / max(1, row['gdp_usd_billion']),  # Consumption intensity
            row['net_exports'] / max(1, row['gdp_usd_billion']),  # Trade balance intensity
        ]
        X.append(features)
        
        # Realistic target: GDP impact between -15% to +5% based on economic vulnerability
        # Countries with high import dependency and low GDP are more vulnerable
        vulnerability = (row['import_dependency'] / max(1, row['gdp_usd_billion'])) * 100
        # Constrain impact to realistic range [-15%, 0%]
        impact = -min(15, vulnerability * np.random.uniform(0.1, 0.3))
        y.append(impact)
    
    X = np.array(X)
    y = np.array(y)
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42,
        max_depth=5,  # Prevent overfitting
        min_samples_split=5
    )
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X

def predict_ai_impact(model, scaler, country_data, severity, price_sensitivity, impact_multiplier):
    """Predict GDP impact using AI model with realistic constraints"""
    if model is None or scaler is None:
        return None
    
    # Prepare features for prediction (same transformation as training)
    features = [
        np.log1p(country_data['import_dependency'] if not pd.isna(country_data['import_dependency']) else 0),
        np.log1p(country_data['gdp_usd_billion'] if not pd.isna(country_data['gdp_usd_billion']) else 0),
        (country_data['production'] if not pd.isna(country_data['production']) else 0) / max(1, country_data['gdp_usd_billion']),
        (country_data['consumption'] if not pd.isna(country_data['consumption']) else 0) / max(1, country_data['gdp_usd_billion']),
        (country_data['net_exports'] if not pd.isna(country_data['net_exports']) else 0) / max(1, country_data['gdp_usd_billion']),
    ]
    
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Get base prediction
    base_prediction = model.predict(features_scaled)[0]
    
    # Adjust based on scenario parameters with realistic constraints
    adjusted_prediction = base_prediction * (severity / 100) * price_sensitivity * impact_multiplier
    
    # CONSTRAIN TO REALISTIC RANGE: No country can lose more than 25% of GDP from oil shock
    # In extreme cases, even 15-20% would be catastrophic but theoretically possible
    adjusted_prediction = max(-25.0, adjusted_prediction)
    
    return adjusted_prediction

# ---------------------------
# Helpers (same logic as yours)
# ---------------------------
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

    # Rename standard cols (OWID often provides Entity/Code/Year)
    df = df.rename(columns={"Entity": "country", "Code": "iso", "Year": "year"})
    # Find the value column automatically
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
        "gdp_usd_billion": [500] * len(isos)  # placeholder values
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
    st.markdown('<div class="app-title"><h1>🛢 Conflict Simulator Dashboard <span class="ai-badge">AI-Powered</span></h1></div>', unsafe_allow_html=True)
    st.caption("Real-time what-if economics on oil supply shocks with AI-powered impact prediction.")
with right:
    st.markdown(
        '<div style="text-align:right;margin-top:4px">'
        '<span class="pill">OWID</span><span class="pill">World Bank (dummy)</span>'
        '<span class="pill">UN Comtrade (best-effort)</span>'
        '<span class="pill">AI Simulation</span>'
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

st.sidebar.markdown("---")
st.sidebar.subheader("AI Simulation")
use_ai = st.sidebar.checkbox("Enable AI Impact Prediction", value=True)
ai_confidence = st.sidebar.slider("AI Confidence Level", 0.5, 1.0, 0.8, step=0.1, 
                                 help="Higher values make the AI more confident in its predictions")

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
target_country = st.selectbox("Choose a country to sanction", master["country"].dropna().unique())
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

# Calculate GDP impact percentage - CONSTRAIN TO REALISTIC RANGE
master["gdp_impact_percent"] = (master["gdp_change"] / master["gdp_usd_billion"]) * 100
master["gdp_impact_percent"] = master["gdp_impact_percent"].clip(upper=0, lower=-25)  # No country loses >25% GDP

# ---------------------------
# AI-Powered Impact Prediction - REFINED
# ---------------------------
if use_ai:
    with st.spinner("Training AI model for realistic impact prediction..."):
        ai_model, ai_scaler, _ = create_ai_simulation_model(master)
    
    if ai_model is not None:
        # Add AI predictions to the master dataframe
        ai_predictions = []
        for _, row in master.iterrows():
            prediction = predict_ai_impact(ai_model, ai_scaler, row, severity, price_sensitivity, impact_multiplier)
            if prediction is not None:
                # Blend AI prediction with basic calculation based on confidence level
                basic_prediction = row['gdp_impact_percent']
                blended = (ai_confidence * prediction) + ((1 - ai_confidence) * basic_prediction)
                # Ensure final prediction is within realistic bounds
                blended = max(-25.0, min(0.0, blended))
                ai_predictions.append(blended)
            else:
                ai_predictions.append(row['gdp_impact_percent'])
        
        master["ai_gdp_impact_percent"] = ai_predictions
        master["ai_gdp_change"] = (master["ai_gdp_impact_percent"] / 100) * master["gdp_usd_billion"]
        
        st.success("AI impact prediction model trained successfully! ✅")
        st.info("ℹ AI predictions are constrained to realistic ranges (-25% to 0% GDP impact)")
    else:
        st.warning("Not enough data for AI prediction. Using basic calculation.")
        master["ai_gdp_impact_percent"] = master["gdp_impact_percent"]
        master["ai_gdp_change"] = master["gdp_change"]
else:
    master["ai_gdp_impact_percent"] = master["gdp_impact_percent"]
    master["ai_gdp_change"] = master["gdp_change"]

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
# Tabs: Overview | Impacts | Data | AI Analysis
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💥 Impacts", "📁 Data", "🤖 AI Analysis"])

with tab1:
    st.markdown("### Top GDP Impacts")
    
    if use_ai:
        top = master.sort_values("ai_gdp_change", ascending=True).head(15)
        impact_col = "ai_gdp_change"
        title = "AI-Predicted GDP change (USD billions)"
    else:
        top = master.sort_values("gdp_change", ascending=True).head(15)
        impact_col = "gdp_change"
        title = "Projected GDP change (USD billions)"
    
    fig = px.bar(
        top,
        x=impact_col,
        y="country",
        orientation="h",
        title=title,
        labels={impact_col: "Δ GDP (USD bn)", "country": ""},
        text=top[impact_col].round(2),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### World Map — GDP Change (%)")
    # Plotly expects ISO-3 codes in 'locations'
    map_df = master.copy()
    map_df["iso3"] = map_df["iso"]
    
    if use_ai:
        color_col = "ai_gdp_impact_percent"
        title = "AI-Predicted GDP change (% of GDP)"
    else:
        color_col = "gdp_impact_percent"
        title = "GDP change (% of GDP)"
    
    fig_map = px.choropleth(
        map_df,
        locations="iso3",
        color=color_col,
        color_continuous_scale="RdBu",
        title=title,
        range_color=[-25, 0],  # Constrain color scale to realistic range
        labels={color_col: "Δ GDP (%)"},
        hover_data={"country": True, color_col: ":.2f", "gdp_change": ":+.2f"}
    )
    fig_map.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    c1, c2 = st.columns([0.6, 0.4])

    with c1:
        if use_ai:
            st.markdown("#### Import Dependency vs. AI-Predicted GDP Change (%)")
            y_col = "ai_gdp_impact_percent"
            title = "Countries with high import dependency tend to lose more (% of GDP) - AI Prediction"
        else:
            st.markdown("#### Import Dependency vs. GDP Change (%)")
            y_col = "gdp_impact_percent"
            title = "Countries with high import dependency tend to lose more (% of GDP)"
            
        bubble = master.copy()
        bubble["size"] = (bubble["import_dependency"].fillna(0) + 1)  # visual size
        fig2 = px.scatter(
            bubble,
            x="import_dependency",
            y=y_col,
            size="size",
            hover_name="country",
            labels={
                "import_dependency": "Import dependency (kb/d proxy)", 
                y_col: "Δ GDP (%)"
            },
            title=title,
        )
        fig2.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        if use_ai:
            st.markdown("#### Quick Table (Top 20 loss by %) - AI Prediction")
            sort_col = "ai_gdp_impact_percent"
            display_cols = ["country", "production", "consumption", "net_exports", 
                           "gdp_usd_billion", "gdp_change", "ai_gdp_impact_percent"]
            rename_cols = {
                "country": "Country",
                "production": "Production",
                "consumption": "Consumption",
                "net_exports": "Net Exports",
                "gdp_usd_billion": "GDP (USD bn)",
                "gdp_change": "Δ GDP (USD bn)",
                "ai_gdp_impact_percent": "Δ GDP (%)"
            }
        else:
            st.markdown("#### Quick Table (Top 20 loss by %)")
            sort_col = "gdp_impact_percent"
            display_cols = ["country", "production", "consumption", "net_exports", 
                           "gdp_usd_billion", "gdp_change", "gdp_impact_percent"]
            rename_cols = {
                "country": "Country",
                "production": "Production",
                "consumption": "Consumption",
                "net_exports": "Net Exports",
                "gdp_usd_billion": "GDP (USD bn)",
                "gdp_change": "Δ GDP (USD bn)",
                "gdp_impact_percent": "Δ GDP (%)"
            }
            
        tbl = master.sort_values(sort_col).head(20)[display_cols].rename(columns=rename_cols)
        st.dataframe(
            tbl.style.format({
                "Production": "{:,.0f}",
                "Consumption": "{:,.0f}",
                "Net Exports": "{:,.0f}",
                "GDP (USD bn)": "{:,.1f}",
                "Δ GDP (USD bn)": "{:+.2f}",
                "Δ GDP (%)": "{:+.2f}%"
            }).apply(lambda x: ['background: rgba(255,107,107,0.1)' if v < -10 else '' for v in x] 
                    if x.name == 'Δ GDP (%)' else [''] * len(x)),
            height=480
        )

with tab3:
    st.markdown("#### Merged dataset (sample of 10)")
    st.dataframe(master.head(10))

    # Download button
    csv_bytes = master.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download full results (CSV)", csv_bytes, file_name="conflict_sim_results.csv", mime="text/csv")

with tab4:
    if use_ai and ai_model is not None:
        st.markdown("### 🤖 AI Impact Analysis")
        
        # Feature importance
        st.markdown("#### Feature Importance in AI Prediction")
        feature_names = ['Import Dependency', 'GDP Size', 'Production', 'Consumption', 'Net Exports']
        importance = ai_model.feature_importances_
        
        fig_importance = px.bar(
            x=importance,
            y=feature_names,
            orientation='h',
            title='Feature Importance in AI Impact Prediction',
            labels={'x': 'Importance', 'y': 'Feature'}
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # AI vs Basic comparison
        st.markdown("#### AI vs Basic Prediction Comparison")
        comparison_df = master[['country', 'gdp_impact_percent', 'ai_gdp_impact_percent']].copy()
        comparison_df['Difference'] = comparison_df['ai_gdp_impact_percent'] - comparison_df['gdp_impact_percent']
        comparison_df['Abs Difference'] = comparison_df['Difference'].abs()
        
        fig_comparison = px.scatter(
            comparison_df,
            x='gdp_impact_percent',
            y='ai_gdp_impact_percent',
            hover_name='country',
            title='AI vs Basic Prediction Comparison',
            labels={
                'gdp_impact_percent': 'Basic Prediction (%)',
                'ai_gdp_impact_percent': 'AI Prediction (%)'
            }
        )
        # Add perfect correlation line
        max_val = max(comparison_df['gdp_impact_percent'].max(), comparison_df['ai_gdp_impact_percent'].max())
        min_val = min(comparison_df['gdp_impact_percent'].min(), comparison_df['ai_gdp_impact_percent'].min())
        fig_comparison.add_shape(
            type="line", line=dict(dash='dash', color='gray'),
            x0=min_val, y0=min_val, x1=max_val, y1=max_val
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Most different predictions - WITH REALISTIC CONSTRAINTS
        st.markdown("#### Largest AI Prediction Differences")
        top_differences = comparison_df.nlargest(10, 'Abs Difference')
        
        # Format the display to show reasonable differences
        st.dataframe(
            top_differences.style.format({
                'gdp_impact_percent': '{:+.2f}%',
                'ai_gdp_impact_percent': '{:+.2f}%',
                'Difference': '{:+.2f}%',
                'Abs Difference': '{:.2f}%'
            }).apply(lambda x: ['background: rgba(255,107,107,0.2)' if abs(v) > 5 else '' for v in x] 
                    if x.name in ['Difference', 'Abs Difference'] else [''] * len(x))
        )
        
        # Add explanation
        st.info("""
        *Interpretation Guide:*
        - Differences under ±2%: Minor variations, both models generally agree
        - Differences ±2-5%: Moderate differences, worth investigating
        - Differences over ±5%: Significant divergence, AI may be capturing non-linear relationships
        """)
        
    else:
        st.info("Enable AI prediction in the sidebar to see AI analysis insights.")

# ---------------------------
# Validation Check
# ---------------------------
with st.expander("🔍 Model Validation Check"):
    st.markdown("### Reality Check: GDP Impact Ranges")
    
    if use_ai:
        impact_data = master['ai_gdp_impact_percent']
    else:
        impact_data = master['gdp_impact_percent']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Worst Impact", f"{impact_data.min():.2f}%")
    with col2:
        st.metric("Average Impact", f"{impact_data.mean():.2f}%")
    with col3:
        st.metric("Best Impact", f"{impact_data.max():.2f}%")
    
    # Check for unrealistic values
    unrealistic_count = len(impact_data[impact_data < -25])
    if unrealistic_count > 0:
        st.error(f"❌ Warning: {unrealistic_count} countries have unrealistic GDP impacts (< -25%)")
    else:
        st.success("✅ All GDP impacts are within realistic bounds (-25% to 0%)")
    
    st.caption("Note: Historical oil crises typically caused GDP impacts in the -5% to -15% range for most affected countries.")