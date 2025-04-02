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
from pages import dashboard, supply_demand_explorer, forecasting, model_insights

# Set page configuration
st.set_page_config(
    page_title="OilX Market Analyzer",
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
try:
    st.sidebar.image("logo.png", width=100)
except:
    st.sidebar.info("Logo file not found.")

# Navigation
page = st.sidebar.radio("Navigation", 
    ["Dashboard", "Supply & Demand Explorer", "Forecasting", 
     "Model Insights", "Scenario Analysis", "Correlation Analysis"])

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

# Add simplified filters to sidebar
st.sidebar.markdown("### Quick Filters")

# Create a more efficient filter system
selected_products = st.sidebar.multiselect(
    "Products", 
    products, 
    default=[products[0] if products else None]
)

# Instead of filtering countries in the sidebar, we'll do that in the dashboard
# This keeps the sidebar cleaner and more focused

# Date range filter with improved defaults
min_date = pd.to_datetime(data["ReferenceDate"]).min().date()
max_date = pd.to_datetime(data["ReferenceDate"]).max().date()

# Default to last 2 years
default_start = max_date - timedelta(days=365*2)
if default_start < min_date:
    default_start = min_date

date_range = st.sidebar.date_input(
    "Date Range",
    value=(default_start, max_date),
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

# Apply product filter if selected
if selected_products:
    filtered_data = filtered_data[filtered_data["Product"].isin(selected_products)]

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
try:
    st.sidebar.markdown(f"v{config.APP_VERSION}")
except:
    st.sidebar.markdown("v1.0.0")

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
    # fallback for pages not yet implemented
    st.title("Scenario Analysis")
    st.info("This feature is under development.")
elif page == "Correlation Analysis":
    # fallback for pages not yet implemented
    st.title("Correlation Analysis")
    st.info("This feature is under development.")
