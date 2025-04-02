import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import os

def load_data():
    """
    Load and preprocess supply and demand data
    
    Returns:
    --------
    data : pandas DataFrame
        Preprocessed data ready for analysis
    """
    try:
        # Define path to CSV file - use a relative path based on app location
        file_path = "SupplyDemand_TopOilCountries_Under24MB.csv"
        
        # Check if file exists
        if not os.path.exists(file_path):
            st.error(f"Data file not found: {file_path}")
            return pd.DataFrame()
        
        # Load the data with appropriate dtypes to reduce memory usage
        dtypes = {
            'CountryISOCode': 'str',
            'CountryName': 'str',
            'FlowBreakdown': 'str',
            'Product': 'str',
            'UnitMeasure': 'str',
            'GeneralizedSource': 'str',
            'RunDate': 'str'
        }
        
        data = pd.read_csv(file_path, dtype=dtypes)
        
        # Convert dates
        data['ReferenceDate'] = pd.to_datetime(data['ReferenceDate'])
        data['RunDate'] = pd.to_datetime(data['RunDate'])
        
        # Basic data validation
        if data.empty:
            st.error("Data file is empty")
            return pd.DataFrame()
        
        # Log some basic statistics
        st.sidebar.markdown(f"**Data Statistics:**")
        st.sidebar.markdown(f"- Rows: {len(data):,}")
        st.sidebar.markdown(f"- Countries: {data['CountryName'].nunique()}")
        st.sidebar.markdown(f"- Products: {data['Product'].nunique()}")
        st.sidebar.markdown(f"- Flow Types: {data['FlowBreakdown'].nunique()}")
        st.sidebar.markdown(f"- Date Range: {data['ReferenceDate'].min().date()} to {data['ReferenceDate'].max().date()}")
        
        return data
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def preprocess_data(data):
    """
    Apply data preprocessing and cleaning steps
    
    Parameters:
    -----------
    data : pandas DataFrame
        Raw data from load_data()
        
    Returns:
    --------
    processed_data : pandas DataFrame
        Cleaned and processed data
    """
    if data.empty:
        return pd.DataFrame()
    
    # Make a copy to avoid modifying the original data
    processed_data = data.copy()
    
    # Filter out zero or negative values that might represent missing data
    processed_data = processed_data[processed_data['ObservedValue'] > 0]
    
    # Handle missing values
    processed_data['ObservedValue'] = processed_data['ObservedValue'].fillna(0)
    
    # Additional preprocessing steps
    
    # 1. Create a month and year column for easier grouping
    processed_data['Year'] = processed_data['ReferenceDate'].dt.year
    processed_data['Month'] = processed_data['ReferenceDate'].dt.month
    processed_data['Quarter'] = processed_data['ReferenceDate'].dt.quarter
    
    # 2. Standardize units if needed
    # Example: Convert to a standard unit if multiple units exist
    if processed_data['UnitMeasure'].nunique() > 1:
        # This is just a placeholder - actual conversion would depend on your data
        st.warning("Multiple units detected. Unit conversion may be needed.")
    
    # 3. Create derived metrics if needed
    # Examples:
    # - Calculate net imports (imports - exports)
    # - Calculate supply balance
    
    # If needed, add additional derived metrics here
    
    # 4. Identify anomalies or outliers
    # Simple example: flag values that are >3 standard deviations from the mean
    for country in processed_data['CountryName'].unique():
        for product in processed_data['Product'].unique():
            for flow in processed_data['FlowBreakdown'].unique():
                # Get data subset
                subset = processed_data[(processed_data['CountryName'] == country) & 
                                        (processed_data['Product'] == product) &
                                        (processed_data['FlowBreakdown'] == flow)]
                
                if len(subset) > 10:  # Only if we have enough data points
                    # Calculate mean and std
                    mean_val = subset['ObservedValue'].mean()
                    std_val = subset['ObservedValue'].std()
                    
                    # Flag outliers
                    outlier_indices = subset[abs(subset['ObservedValue'] - mean_val) > 3 * std_val].index
                    
                    # Add outlier flag
                    processed_data.loc[outlier_indices, 'IsOutlier'] = True
    
    # Fill missing outlier flags with False
    processed_data['IsOutlier'] = processed_data.get('IsOutlier', False)
    
    # Return the processed data
    return processed_data

def aggregate_data(data, groupby_cols, agg_col='ObservedValue', agg_func='sum'):
    """
    Aggregate data by specified columns
    
    Parameters:
    -----------
    data : pandas DataFrame
        Data to aggregate
    groupby_cols : list
        Columns to group by
    agg_col : str
        Column to aggregate
    agg_func : str or dict
        Aggregation function(s) to apply
        
    Returns:
    --------
    aggregated_data : pandas DataFrame
        Aggregated data
    """
    if data.empty:
        return pd.DataFrame()
    
    # Check if all groupby columns exist
    missing_cols = [col for col in groupby_cols if col not in data.columns]
    if missing_cols:
        st.warning(f"Missing columns for groupby: {missing_cols}")
        # Use only available columns
        groupby_cols = [col for col in groupby_cols if col in data.columns]
        
        if not groupby_cols:
            st.error("No valid columns for grouping")
            return pd.DataFrame()
    
    # Perform aggregation
    try:
        agg_data = data.groupby(groupby_cols)[agg_col].agg(agg_func).reset_index()
        return agg_data
    except Exception as e:
        st.error(f"Error during aggregation: {e}")
        return pd.DataFrame()

def get_time_series(data, country, product, flow_breakdown, freq='M'):
    """
    Extract a time series for a specific country, product, and flow
    
    Parameters:
    -----------
    data : pandas DataFrame
        Data containing time series
    country : str
        Country name
    product : str
        Product name
    flow_breakdown : str
        Flow breakdown category
    freq : str
        Time frequency ('M' for monthly, 'Q' for quarterly, 'Y' for yearly)
        
    Returns:
    --------
    time_series : pandas Series
        Time series data with datetime index
    """
    if data.empty:
        return pd.Series()
    
    # Filter data for the specified parameters
    filtered_data = data[
        (data['CountryName'] == country) &
        (data['Product'] == product) &
        (data['FlowBreakdown'] == flow_breakdown)
    ].copy()
    
    if filtered_data.empty:
        st.warning(f"No data found for Country: {country}, Product: {product}, Flow: {flow_breakdown}")
        return pd.Series()
    
    # Determine the appropriate date component for grouping
    if freq == 'M':
        # Group by year and month
        filtered_data['Period'] = filtered_data['ReferenceDate'].dt.to_period('M')
    elif freq == 'Q':
        # Group by year and quarter
        filtered_data['Period'] = filtered_data['ReferenceDate'].dt.to_period('Q')
    elif freq == 'Y':
        # Group by year
        filtered_data['Period'] = filtered_data['ReferenceDate'].dt.to_period('Y')
    else:
        st.warning(f"Unsupported frequency: {freq}. Using monthly.")
        filtered_data['Period'] = filtered_data['ReferenceDate'].dt.to_period('M')
    
    # Group by period and aggregate
    time_series = filtered_data.groupby('Period')['ObservedValue'].mean()
    
    # Convert period index to datetime for compatibility with forecasting functions
    time_series.index = time_series.index.to_timestamp()
    
    # Sort by date
    time_series = time_series.sort_index()
    
    return time_series

def get_comparable_flows(data, country, product, reference_flow=None):
    """
    Get flows that can be compared for a given country and product
    
    Parameters:
    -----------
    data : pandas DataFrame
        Data containing flows
    country : str
        Country name
    product : str
        Product name
    reference_flow : str, optional
        Reference flow to find related flows
        
    Returns:
    --------
    comparable_flows : list
        List of flows that can be compared
    """
    if data.empty:
        return []
    
    # Filter data for the specified parameters
    filtered_data = data[
        (data['CountryName'] == country) &
        (data['Product'] == product)
    ]
    
    if filtered_data.empty:
        return []
    
    # Get all available flows
    all_flows = filtered_data['FlowBreakdown'].unique().tolist()
    
    # If reference flow is provided, try to find related flows
    if reference_flow:
        # For example, if reference flow contains "Production",
        # find other flows related to production
        if "Production" in reference_flow:
            comparable_flows = [flow for flow in all_flows if "Production" in flow or "Consumption" in flow]
        elif "Import" in reference_flow:
            comparable_flows = [flow for flow in all_flows if "Import" in flow or "Export" in flow]
        elif "Consumption" in reference_flow:
            comparable_flows = [flow for flow in all_flows if "Consumption" in flow or "Production" in flow]
        else:
            comparable_flows = all_flows
    else:
        comparable_flows = all_flows
    
    return comparable_flows

def check_data_quality(data):
    """
    Check data quality and report issues
    
    Parameters:
    -----------
    data : pandas DataFrame
        Data to check
        
    Returns:
    --------
    quality_report : dict
        Report of data quality issues
    """
    quality_report = {
        "missing_values": {},
        "outliers": {},
        "duplicates": 0,
        "date_coverage": {},
        "completeness": {}
    }
    
    if data.empty:
        return quality_report
    
    # Check missing values
    for col in data.columns:
        missing_count = data[col].isna().sum()
        if missing_count > 0:
            quality_report["missing_values"][col] = {
                "count": missing_count,
                "percentage": (missing_count / len(data)) * 100
            }
    
    # Check outliers (simple Z-score method)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
        outliers = len(z_scores[z_scores > 3])
        if outliers > 0:
            quality_report["outliers"][col] = {
                "count": outliers,
                "percentage": (outliers / len(data)) * 100
            }
    
    # Check duplicates
    duplicates = data.duplicated().sum()
    quality_report["duplicates"] = duplicates
    
    # Check date coverage
    if 'ReferenceDate' in data.columns:
        # Get date range
        min_date = data['ReferenceDate'].min()
        max_date = data['ReferenceDate'].max()
        
        # Calculate expected number of periods
        days = (max_date - min_date).days
        
        # Assuming monthly data
        expected_months = days // 30
        actual_months = len(data['ReferenceDate'].dt.to_period('M').unique())
        
        quality_report["date_coverage"] = {
            "min_date": min_date,
            "max_date": max_date,
            "expected_periods": expected_months,
            "actual_periods": actual_months,
            "coverage_percentage": (actual_months / expected_months * 100) if expected_months > 0 else 0
        }
    
    # Check completeness by country, product, flow
    if all(col in data.columns for col in ['CountryName', 'Product', 'FlowBreakdown']):
        # For each country-product-flow combination, check how complete the time series is
        for country in data['CountryName'].unique():
            for product in data['Product'].unique():
                for flow in data['FlowBreakdown'].unique():
                    # Get the data subset
                    subset = data[
                        (data['CountryName'] == country) &
                        (data['Product'] == product) &
                        (data['FlowBreakdown'] == flow)
                    ]
                    
                    if not subset.empty:
                        # Get date range
                        min_date = subset['ReferenceDate'].min()
                        max_date = subset['ReferenceDate'].max()
                        
                        # Calculate expected number of periods (monthly)
                        expected_periods = ((max_date.year - min_date.year) * 12 + 
                                          max_date.month - min_date.month + 1)
                        
                        # Count actual periods
                        actual_periods = subset['ReferenceDate'].dt.to_period('M').nunique()
                        
                        # Calculate completeness
                        completeness = (actual_periods / expected_periods * 100) if expected_periods > 0 else 0
                        
                        # Add to report if completeness is below threshold
                        if completeness < 90:
                            key = f"{country} - {product} - {flow}"
                            quality_report["completeness"][key] = {
                                "expected_periods": expected_periods,
                                "actual_periods": actual_periods,
                                "completeness_percentage": completeness
                            }
    
    return quality_report