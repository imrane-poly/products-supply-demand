import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Import modules
from modules.data_loader import load_data, preprocess_data
from modules.utils import configure_page, session_state_init
import config

# Page imports
from pages import dashboard, supply_demand_explorer, forecasting, model_insights#, scenario_analysis, correlation_analysis

# Set page configuration
st.set_page_config(
    page_title="OilX Supply & Demand Forecaster",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
configure_page()

# Initialize session state
session_state_init()

# Load data
@st.cache_data(ttl=3600)
def get_data():
    """Load and cache data"""
    data = load_data()
    return data

# Sidebar navigation
st.sidebar.markdown('<div class="main-header">OilX Market Analyzer</div>', unsafe_allow_html=True)

# Add logo (comment out if not available)
st.sidebar.image("logo.png", width=100)

# Navigation
page = st.sidebar.radio("Navigation", 
    ["Dashboard", "Supply & Demand Explorer", "Forecasting", 
     "Model Insights", "Scenario Analysis", "Correlation Analysis"])

# Data filters
st.sidebar.markdown("### Data Filters")

# Load base data
with st.spinner("Loading data..."):
    data = get_data()

if data.empty:
    st.error("Failed to load data. Please check your data file.")
    st.stop()

# Extract unique values for filters
countries = sorted(data["CountryName"].unique())
products = sorted(data["Product"].unique())
flow_metrics = sorted(data["FlowBreakdown"].unique())

# Add filters to sidebar
selected_countries = st.sidebar.multiselect("Countries", countries, default=countries[:3])
selected_products = st.sidebar.multiselect("Products", products, default=[products[0]])
selected_flows = st.sidebar.multiselect("Flow Metrics", flow_metrics, default=[f for f in flow_metrics if "Demand" in f][:1])

# Date range filter
min_date = pd.to_datetime(data["ReferenceDate"]).min().date()
max_date = pd.to_datetime(data["ReferenceDate"]).max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(max_date - timedelta(days=365*2), max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = min_date
    end_date = max_date

# Filter data
filtered_data = data.copy()

# Apply date filter
filtered_data = filtered_data[
    (pd.to_datetime(filtered_data["ReferenceDate"]).dt.date >= start_date) & 
    (pd.to_datetime(filtered_data["ReferenceDate"]).dt.date <= end_date)
]

# Apply other filters if selected
if selected_countries:
    filtered_data = filtered_data[filtered_data["CountryName"].isin(selected_countries)]

if selected_products:
    filtered_data = filtered_data[filtered_data["Product"].isin(selected_products)]

if selected_flows:
    filtered_data = filtered_data[filtered_data["FlowBreakdown"].isin(selected_flows)]

# Process the data
processed_data = preprocess_data(filtered_data)

# Show data stats in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Data Statistics:**")
st.sidebar.markdown(f"- Records: {len(processed_data):,}")
st.sidebar.markdown(f"- Countries: {processed_data['CountryName'].nunique()}")
st.sidebar.markdown(f"- Products: {processed_data['Product'].nunique()}")
st.sidebar.markdown(f"- Date Range: {processed_data['ReferenceDate'].min().date()} to {processed_data['ReferenceDate'].max().date()}")

# Version info
st.sidebar.markdown("---")
st.sidebar.markdown(f"v{config.APP_VERSION}")

# Load the appropriate page
if page == "Dashboard":
    dashboard.show(processed_data)
elif page == "Supply & Demand Explorer":
    supply_demand_explorer.show(processed_data)
elif page == "Forecasting":
    forecasting.show(processed_data)
elif page == "Model Insights":
    model_insights.show(processed_data)
elif page == "Scenario Analysis":
    scenario_analysis.show(processed_data)
elif page == "Correlation Analysis":
    correlation_analysis.show(processed_data)