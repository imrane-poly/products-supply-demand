import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
import string
import re
import os
import json
import config

def configure_page():
    """Configure Streamlit page with custom CSS and settings"""
    # Add custom CSS
    st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)
    
    # Set wide layout
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Additional page configuration options can be added here

def session_state_init():
    """Initialize session state variables"""
    # Initialize session state variables if they don't exist
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.model_cache = {}
        st.session_state.forecast_history = []
        st.session_state.selected_features = []
        st.session_state.custom_scenarios = {}
        st.session_state.comparison_models = []

def generate_id(prefix="", length=8):
    """Generate a random ID for objects"""
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{prefix}_{random_part}" if prefix else random_part

def format_timestamp(dt=None):
    """Format a timestamp for filenames"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%dT%H-%M-%S")

def parse_date_range(date_range):
    """Parse a date range from Streamlit date_input"""
    if len(date_range) == 2:
        start_date, end_date = date_range
        return start_date, end_date
    elif len(date_range) == 1:
        return date_range[0], date_range[0]
    else:
        # Default to last year if no date range is provided
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)
        return start_date, end_date

def get_time_freq(aggregation):
    """Get pandas time frequency string based on aggregation level"""
    if aggregation.upper() == "MONTHLY":
        return "MS"
    elif aggregation.upper() == "QUARTERLY":
        return "QS"
    elif aggregation.upper() == "YEARLY":
        return "YS"
    else:
        return "MS"  # Default to monthly

def filter_dataframe(df, filters):
    """
    Filter a DataFrame based on a dictionary of filters
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame to filter
    filters : dict
        Dictionary of filters with format {column: value or list of values}
        
    Returns:
    --------
    filtered_df : pandas DataFrame
        Filtered DataFrame
    """
    if filters is None:
        return df
    
    filtered_df = df.copy()
    
    for col, values in filters.items():
        if col in filtered_df.columns:
            if isinstance(values, list):
                filtered_df = filtered_df[filtered_df[col].isin(values)]
            else:
                filtered_df = filtered_df[filtered_df[col] == values]
    
    return filtered_df

def format_number(num, precision=2):
    """Format a number with thousands separator and fixed precision"""
    if isinstance(num, (int, float)):
        if abs(num) >= 1e6:
            return f"{num / 1e6:.{precision}f}M"
        elif abs(num) >= 1e3:
            return f"{num / 1e3:.{precision}f}K"
        else:
            return f"{num:.{precision}f}"
    return str(num)

def percentage_change(current, previous):
    """Calculate percentage change between two values"""
    if previous == 0:
        return float('inf') if current > 0 else float('-inf') if current < 0 else 0
    
    return ((current - previous) / abs(previous)) * 100

def calculate_growth_rate(series, periods=1):
    """Calculate compound annual growth rate for a time series"""
    if len(series) <= periods:
        return None
    
    start_value = series.iloc[0]
    end_value = series.iloc[-1]
    
    if start_value <= 0:
        return None
    
    n_periods = len(series) - 1
    
    # Formula: (end/start)^(1/periods) - 1
    return (pow(end_value / start_value, 1 / n_periods) - 1) * 100

def moving_average(series, window=3):
    """Calculate moving average for a series"""
    return series.rolling(window=window, min_periods=1).mean()

def exponential_smoothing(series, alpha=0.3):
    """Apply exponential smoothing to a series"""
    return series.ewm(alpha=alpha, adjust=False).mean()

def detect_outliers(series, threshold=3.0):
    """
    Detect outliers in a series using Z-score method
    
    Parameters:
    -----------
    series : pandas Series
        Series to check for outliers
    threshold : float
        Z-score threshold for outlier detection
        
    Returns:
    --------
    outliers : pandas Series
        Boolean series indicating outlier positions
    """
    if len(series) < 4:  # Need at least a few points for meaningful statistics
        return pd.Series(False, index=series.index)
    
    # Calculate Z-scores
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        return pd.Series(False, index=series.index)
    
    z_scores = (series - mean) / std
    
    # Identify outliers
    outliers = abs(z_scores) > threshold
    
    return outliers

def impute_missing_values(series, method='linear'):
    """
    Impute missing values in a time series
    
    Parameters:
    -----------
    series : pandas Series
        Time series with missing values
    method : str
        Imputation method ('linear', 'ffill', 'bfill', 'mean', 'median')
        
    Returns:
    --------
    imputed_series : pandas Series
        Series with imputed values
    """
    if method == 'linear':
        return series.interpolate(method='linear')
    elif method == 'ffill':
        return series.ffill()
    elif method == 'bfill':
        return series.bfill()
    elif method == 'mean':
        return series.fillna(series.mean())
    elif method == 'median':
        return series.fillna(series.median())
    else:
        # Default to linear interpolation
        return series.interpolate(method='linear')

def resample_time_series(series, freq='MS'):
    """
    Resample a time series to a different frequency
    
    Parameters:
    -----------
    series : pandas Series
        Time series to resample
    freq : str
        Target frequency (pandas frequency string)
        
    Returns:
    --------
    resampled_series : pandas Series
        Resampled time series
    """
    # Make sure the series has a datetime index
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex for resampling")
    
    # Resample
    resampled = series.resample(freq).mean()
    
    # Handle missing values
    resampled = impute_missing_values(resampled, method='linear')
    
    return resampled

def get_trend_direction(series, lookback=3):
    """
    Get the trend direction for a time series
    
    Parameters:
    -----------
    series : pandas Series
        Time series to analyze
    lookback : int
        Number of periods to look back
        
    Returns:
    --------
    trend : str
        'up', 'down', or 'flat'
    """
    if len(series) < lookback + 1:
        return 'unknown'
    
    # Get recent values
    recent = series.iloc[-lookback-1:]
    
    # Calculate slope of linear regression
    x = np.arange(len(recent))
    y = recent.values
    slope, _ = np.polyfit(x, y, 1)
    
    # Determine trend direction
    if slope > 0.01 * recent.mean():
        return 'up'
    elif slope < -0.01 * recent.mean():
        return 'down'
    else:
        return 'flat'

def get_seasonality(series, period=12):
    """
    Calculate seasonal factors for a time series
    
    Parameters:
    -----------
    series : pandas Series
        Time series to analyze
    period : int
        Seasonality period (e.g., 12 for monthly data with yearly seasonality)
        
    Returns:
    --------
    seasonal_factors : dict
        Dictionary of seasonal factors by period index
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Only perform decomposition if we have enough data
    if len(series) < 2 * period:
        return {}
    
    try:
        # Perform decomposition
        decomposition = seasonal_decompose(series, model='additive', period=period)
        
        # Get seasonal factors
        seasonal = decomposition.seasonal
        
        # Create a dictionary of factors by period index
        seasonal_factors = {}
        
        for i in range(period):
            # Get all values for this period index
            period_values = [seasonal.iloc[j] for j in range(i, len(seasonal), period)]
            
            # Calculate average factor
            if period_values:
                seasonal_factors[i] = np.mean(period_values)
        
        return seasonal_factors
    
    except Exception as e:
        print(f"Could not calculate seasonality: {e}")
        return {}

def extract_flow_components(flow_breakdown):
    """
    Extract components from a flow breakdown string
    
    Parameters:
    -----------
    flow_breakdown : str
        Flow breakdown string (e.g., "Production - Crude Oil")
        
    Returns:
    --------
    components : dict
        Dictionary of extracted components
    """
    components = {
        "flow_type": None,
        "product": None,
        "source": None,
        "destination": None
    }
    
    # Extract flow type
    flow_types = ["Production", "Consumption", "Imports", "Exports", 
                 "Refinery Input", "Refinery Output", "Stocks", "Stock Change"]
    
    for flow in flow_types:
        if flow in flow_breakdown:
            components["flow_type"] = flow
            break
    
    # Extract product (simplified approach)
    products = ["Crude Oil", "Gasoline", "Diesel", "Jet Fuel", "LPG", 
               "Fuel Oil", "Naphtha", "Natural Gas", "LNG"]
    
    for product in products:
        if product in flow_breakdown:
            components["product"] = product
            break
    
    # Look for source/destination patterns
    from_pattern = r"from\s+(\w+(?:\s+\w+)*)"
    to_pattern = r"to\s+(\w+(?:\s+\w+)*)"
    
    from_match = re.search(from_pattern, flow_breakdown)
    to_match = re.search(to_pattern, flow_breakdown)
    
    if from_match:
        components["source"] = from_match.group(1)
    
    if to_match:
        components["destination"] = to_match.group(1)
    
    return components

def save_forecast_result(forecast_data, model_info, parameters, metadata):
    """
    Save forecast result to session state and optionally to file
    
    Parameters:
    -----------
    forecast_data : pandas DataFrame
        DataFrame containing forecast data
    model_info : dict
        Dictionary with model information
    parameters : dict
        Dictionary of model parameters
    metadata : dict
        Additional metadata about the forecast
    
    Returns:
    --------
    forecast_id : str
        ID of the saved forecast
    """
    # Generate a unique ID for this forecast
    forecast_id = generate_id("forecast")
    
    # Create forecast record
    forecast_record = {
        "id": forecast_id,
        "timestamp": datetime.now().isoformat(),
        "model_type": model_info.get("model_type"),
        "parameters": parameters,
        "metadata": metadata,
        "data": forecast_data.to_dict(orient="records")
    }
    
    # Add to session state history
    if "forecast_history" not in st.session_state:
        st.session_state.forecast_history = []
    
    st.session_state.forecast_history.append(forecast_record)
    
    # Limit history size
    if len(st.session_state.forecast_history) > 10:
        st.session_state.forecast_history = st.session_state.forecast_history[-10:]
    
    # Optionally save to file
    if metadata.get("save_to_file", False):
        try:
            os.makedirs("forecasts", exist_ok=True)
            filename = f"forecasts/forecast_{forecast_id}_{format_timestamp()}.json"
            
            with open(filename, "w") as f:
                json.dump(forecast_record, f, indent=2)
            
            print(f"Forecast saved to {filename}")
        except Exception as e:
            print(f"Error saving forecast to file: {e}")
    
    return forecast_id

def load_forecast(forecast_id):
    """
    Load a forecast from session state or file
    
    Parameters:
    -----------
    forecast_id : str
        ID of the forecast to load
        
    Returns:
    --------
    forecast : dict or None
        Loaded forecast record
    """
    # Check session state first
    if "forecast_history" in st.session_state:
        for forecast in st.session_state.forecast_history:
            if forecast["id"] == forecast_id:
                return forecast
    
    # If not found, try loading from file
    try:
        forecast_files = [f for f in os.listdir("forecasts") if f.startswith(f"forecast_{forecast_id}")]
        
        if forecast_files:
            filename = f"forecasts/{forecast_files[0]}"
            
            with open(filename, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading forecast from file: {e}")
    
    return None

def create_download_link(df, filename, linktext="Download CSV"):
    """
    Create a download link for a DataFrame
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame to download
    filename : str
        Filename for the download
    linktext : str
        Text to display for the download link
        
    Returns:
    --------
    href : str
        HTML for the download link
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{linktext}</a>'
    return href

def display_info_card(title, content, icon=None):
    """
    Display an information card with styled content
    
    Parameters:
    -----------
    title : str
        Card title
    content : str
        Card content (can include HTML)
    icon : str, optional
        Icon to display
    """
    icon_html = f'<i class="material-icons">{icon}</i> ' if icon else ''
    
    st.markdown(f"""
    <div class="info-card">
        <h3>{icon_html}{title}</h3>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)

def display_metric_row(metrics_dict, prefix=""):
    """
    Display a row of metrics using st.columns
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of metrics with format {name: value}
    prefix : str, optional
        Prefix for metric labels
    """
    cols = st.columns(len(metrics_dict))
    
    for i, (name, value) in enumerate(metrics_dict.items()):
        with cols[i]:
            display_metric(name, value, prefix)

def display_metric(label, value, delta=None, delta_color="normal"):
    """
    Display a metric with consistent formatting
    
    Parameters:
    -----------
    label : str
        Metric label
    value : str, int, or float
        Metric value
    delta : str, int, or float, optional
        Change value to display
    delta_color : str
        Color for delta ("normal", "inverse", or "off")
    """
    # Format value if it's a number
    if isinstance(value, (int, float)):
        formatted_value = format_number(value)
    else:
        formatted_value = value
    
    # Format delta if provided
    if delta is not None:
        if isinstance(delta, (int, float)):
            formatted_delta = format_number(delta)
            
            # Add plus sign for positive deltas
            if delta > 0 and not formatted_delta.startswith("+"):
                formatted_delta = f"+{formatted_delta}"
        else:
            formatted_delta = delta
        
        st.metric(label, formatted_value, formatted_delta, delta_color)
    else:
        st.metric(label, formatted_value)

def show_progress_bar(iterable, label="Processing"):
    """
    Show a progress bar for an iterable
    
    Parameters:
    -----------
    iterable : iterable
        Iterable to process
    label : str
        Label for the progress bar
        
    Returns:
    --------
    Generator yielding items from the iterable
    """
    progress_bar = st.progress(0)
    total = len(iterable)
    
    for i, item in enumerate(iterable):
        yield item
        progress_bar.progress((i + 1) / total)
    
    progress_bar.empty()

def timed_execution(func):
    """
    Decorator to measure execution time
    
    Parameters:
    -----------
    func : function
        Function to time
        
    Returns:
    --------
    Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def throttle(wait):
    """
    Decorator to throttle a function
    
    Parameters:
    -----------
    wait : float
        Minimum time between calls (in seconds)
        
    Returns:
    --------
    Decorator function
    """
    def decorator(func):
        last_called = [0]
        
        def throttled(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            
            if elapsed >= wait:
                last_called[0] = time.time()
                return func(*args, **kwargs)
        
        return throttled
    
    return decorator