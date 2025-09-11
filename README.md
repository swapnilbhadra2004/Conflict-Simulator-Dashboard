# Conflict Simulator Dashboard

Implementation for **Conflict Simulator Dashboard** — an interactive tool to simulate economic impacts of commodity supply shocks across countries.

---

## 📄 Table of Contents

- [Overview](#overview)  
- [Demo & Screenshots](#demo--screenshots)  
- [Installation & Setup](#installation--setup)  
- [Usage](#usage)  
- [Controls & Parameters](#controls--parameters)  
- [AI Simulation & Prediction](#ai-simulation--prediction)  
- [Data Sources](#data-sources)  
- [License](#license)  
- [Contact](#contact)  

---

## Overview

This dashboard lets users:

- Choose a commodity (e.g. Oil, Coal, Gas)  
- Set the severity of supply cut (sanctions/shock)  
- View changes in production, consumption, imports/exports  
- Visualize GDP impact across countries using maps & charts  
- (Optional) Use AI-based predictions for more realistic impact estimates  

---

## Demo & Screenshots

> Screenshots coming soon!

![](doc/screenshot_dashboard.png)  
![](doc/img.png)  

---

## Installation & Setup

```bash
# Clone the repo
git clone https://github.com/your-username/conflict-simulator-dashboard.git
cd conflict-simulator-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run conflict_simulator_pretty.py


Usage

Select the commodity from the sidebar.

Choose the year of data.

Adjust the export cut severity, price sensitivity, and impact multiplier.

(Optional) Enable AI Impact Prediction for model-based estimates.

Interact with the maps, charts, and tables to analyze outcomes.


| Parameter                    | Description                                        |
| ---------------------------- | -------------------------------------------------- |
| **Commodity**                | Type of product (Oil, Coal, Gas)                   |
| **Year**                     | Data year used for production/consumption/GDP etc. |
| **Export Cut Severity (%)**  | How large the supply cut is for the target country |
| **Price Sensitivity Factor** | How responsive price is to reduced supply          |
| **Impact Multiplier**        | Scaling factor for GDP impact calculations         |
| **Enable AI Prediction**     | Whether to use ML-based estimate vs formula-based  |


AI Simulation & Prediction

When enabled, the dashboard trains a model using features like:

Import dependency

Production & consumption scaled by GDP

Net exports

Model types used include:

RandomForestRegressor

(Optional) GradientBoostingRegressor

Outputs include:

Predicted GDP impact percentage

Comparison between basic formula vs AI model


Data Sources

Custom CSV dataset (hardcoded_trade_shares_full.csv)

(Previously) OWID, Comtrade, World Bank — now replaced/offline where needed
