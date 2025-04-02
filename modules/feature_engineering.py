import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats

def create_time_features(time_series, periods=[1, 2, 12]):
    """
    Create time-based features from datetime index
    
    Parameters:
    -----------
    time_series : pandas Series
        Time series data with datetime index
        
    Returns:
    --------
    diff_features : pandas DataFrame
        DataFrame containing differenced features
    """
    df = pd.DataFrame(index=time_series.index)
    
    # Create differenced features
    for period in periods:
        if period < len(time_series):  # Only create differences that make sense for the data
            df[f'diff_{period}'] = time_series.diff(periods=period)
            # Add percentage change features
            df[f'pct_change_{period}'] = time_series.pct_change(periods=period)
    
    return df

def create_seasonal_features(time_series, period=12):
    """
    Create seasonal features from time series decomposition
    
    Parameters:
    -----------
    time_series : pandas Series
        Time series data
    period : int
        Seasonality period (e.g., 12 for monthly data with yearly seasonality)
        
    Returns:
    --------
    seasonal_features : pandas DataFrame
        DataFrame containing seasonal features
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    df = pd.DataFrame(index=time_series.index)
    
    # Only perform decomposition if we have enough data
    if len(time_series) >= 2 * period:
        try:
            # Ensure time series is properly indexed for decomposition
            ts = time_series.copy()
            if not isinstance(ts.index, pd.DatetimeIndex):
                raise ValueError("Time series index must be DatetimeIndex for seasonal decomposition")
            
            # Perform decomposition
            decomposition = seasonal_decompose(ts, model='additive', period=period)
            
            # Add components to dataframe
            df['trend'] = decomposition.trend
            df['seasonal'] = decomposition.seasonal
            df['residual'] = decomposition.resid
            
            # Add seasonal strength metric
            seasonal_strength = 1 - (decomposition.resid.var() / (decomposition.seasonal + decomposition.resid).var())
            df['seasonal_strength'] = seasonal_strength
            
        except Exception as e:
            # Handle errors gracefully (e.g., not enough data points)
            print(f"Could not perform seasonal decomposition: {e}")
    
    return df

def create_fourier_features(time_series, period=12, k=2):
    """
    Create Fourier series features for capturing seasonality
    
    Parameters:
    -----------
    time_series : pandas Series
        Time series data
    period : int
        Seasonality period (e.g., 12 for monthly data with yearly seasonality)
    k : int
        Number of Fourier terms to include
        
    Returns:
    --------
    fourier_features : pandas DataFrame
        DataFrame containing Fourier features
    """
    df = pd.DataFrame(index=time_series.index)
    
    # Create time index (normalized within the period)
    n = np.arange(len(time_series))
    
    # Create Fourier terms
    for i in range(1, k+1):
        df[f'fourier_sin_{period}_{i}'] = np.sin(2 * np.pi * i * n / period)
        df[f'fourier_cos_{period}_{i}'] = np.cos(2 * np.pi * i * n / period)
    
    return df

def create_calendar_features(time_series):
    """
    Create calendar-based features
    
    Parameters:
    -----------
    time_series : pandas Series
        Time series data with datetime index
        
    Returns:
    --------
    calendar_features : pandas DataFrame
        DataFrame containing calendar-based features
    """
    dates = time_series.index
    df = pd.DataFrame(index=dates)
    
    # Check if index is DatetimeIndex
    if not isinstance(dates, pd.DatetimeIndex):
        return df
    
    # Holiday features (simplified version)
    df['IsHolidayMonth'] = ((dates.month == 12) | (dates.month == 1)).astype(int)
    df['IsSummerMonth'] = ((dates.month >= 6) & (dates.month <= 8)).astype(int)
    
    # First/last month of quarter
    df['IsFirstMonthOfQuarter'] = ((dates.month % 3) == 1).astype(int)
    df['IsLastMonthOfQuarter'] = ((dates.month % 3) == 0).astype(int)
    
    # First/last month of year
    df['IsFirstMonthOfYear'] = (dates.month == 1).astype(int)
    df['IsLastMonthOfYear'] = (dates.month == 12).astype(int)
    
    return df

def create_interaction_features(features_df, columns=None, degree=2):
    """
    Create interaction features between existing features
    
    Parameters:
    -----------
    features_df : pandas DataFrame
        DataFrame containing features
    columns : list or None
        List of column names to use for interactions. If None, all columns are used.
    degree : int
        Degree of interaction (2 for pairwise, 3 for three-way, etc.)
        
    Returns:
    --------
    interaction_features : pandas DataFrame
        DataFrame containing interaction features
    """
    from itertools import combinations
    
    if columns is None:
        # Use all numeric columns
        columns = features_df.select_dtypes(include=np.number).columns.tolist()
    
    df = pd.DataFrame(index=features_df.index)
    
    # Create interaction features
    for combo in combinations(columns, degree):
        # Create column name
        col_name = f"interaction_{'_'.join(combo)}"
        
        # Create interaction feature
        interaction = features_df[combo[0]].copy()
        for col in combo[1:]:
            interaction = interaction * features_df[col]
        
        df[col_name] = interaction
    
    return df

def create_target_encoding_features(time_series, cat_features_df):
    """
    Create target encoding features for categorical variables
    
    Parameters:
    -----------
    time_series : pandas Series
        Target time series data
    cat_features_df : pandas DataFrame
        DataFrame containing categorical features
        
    Returns:
    --------
    encoded_features : pandas DataFrame
        DataFrame containing target-encoded features
    """
    df = pd.DataFrame(index=time_series.index)
    
    # Process each categorical column
    for col in cat_features_df.columns:
        # Get category values
        categories = cat_features_df[col].unique()
        
        # Calculate mean target value for each category
        encoding_map = {}
        for cat in categories:
            cat_indices = cat_features_df[col] == cat
            if cat_indices.sum() > 0:
                cat_mean = time_series.loc[cat_indices].mean()
                encoding_map[cat] = cat_mean
        
        # Create encoded feature
        df[f'{col}_target_mean'] = cat_features_df[col].map(encoding_map)
    
    return df

def create_feature_set(time_series, include_time_features=True, include_lag_features=True,
                      include_rolling_features=True, include_diff_features=True,
                      include_seasonal_features=True, include_fourier_features=True,
                      max_lag=12, rolling_windows=[3, 6, 12], diff_periods=[1, 12],
                      seasonal_period=12, fourier_terms=3):
    """
    Create a complete set of features for time series forecasting
    
    Parameters:
    -----------
    time_series : pandas Series
        Time series data with datetime index
    include_* : bool
        Whether to include different feature types
    max_lag : int
        Maximum lag period for lagged features
    rolling_windows : list
        List of window sizes for rolling features
    diff_periods : list
        List of periods for differenced features
    seasonal_period : int
        Period for seasonal decomposition and Fourier features
    fourier_terms : int
        Number of Fourier terms to include
        
    Returns:
    --------
    features : pandas DataFrame
        DataFrame containing all generated features
    """
    # Initialize features DataFrame
    features = pd.DataFrame(index=time_series.index)
    
    # Add time features
    if include_time_features:
        time_features = create_time_features(time_series)
        features = pd.concat([features, time_features], axis=1)
    
    # Add lagged features
    if include_lag_features:
        lag_periods = list(range(1, max_lag + 1))
        lagged_features = create_lagged_features(time_series, lags=lag_periods)
        features = pd.concat([features, lagged_features], axis=1)
    
    # Add rolling features
    if include_rolling_features:
        rolling_features = create_rolling_features(time_series, windows=rolling_windows)
        features = pd.concat([features, rolling_features], axis=1)
    
    # Add differenced features
    if include_diff_features:
        diff_features = create_diff_features(time_series, periods=diff_periods)
        features = pd.concat([features, diff_features], axis=1)
    
    # Add seasonal features
    if include_seasonal_features:
        seasonal_features = create_seasonal_features(time_series, period=seasonal_period)
        features = pd.concat([features, seasonal_features], axis=1)
    
    # Add Fourier features
    if include_fourier_features:
        fourier_features = create_fourier_features(time_series, period=seasonal_period, k=fourier_terms)
        features = pd.concat([features, fourier_features], axis=1)
    
    # Drop any duplicate columns (in case of overlapping features)
    features = features.loc[:, ~features.columns.duplicated()]
    
    return features

def create_lagged_features(time_series, lags=[1, 2, 3, 6, 12]):
    """
    Create lagged features from time series
    
    Parameters:
    -----------
    time_series : pandas Series
        Time series data
    lags : list
        List of lag periods to create
        
    Returns:
    --------
    lagged_features : pandas DataFrame
        DataFrame containing lagged features
    """
    df = pd.DataFrame(index=time_series.index)
    
    # Create lagged features
    for lag in lags:
        if lag < len(time_series):  # Only create lags that make sense for the data
            df[f'lag_{lag}'] = time_series.shift(lag)
    
    return df

def create_rolling_features(time_series, windows=[3, 6, 12], functions=['mean', 'std', 'min', 'max']):
    """
    Create rolling window features from time series
    
    Parameters:
    -----------
    time_series : pandas Series
        Time series data
    windows : list
        List of window sizes for rolling calculations
    functions : list
        List of functions to apply to rolling windows
        
    Returns:
    --------
    rolling_features : pandas DataFrame
        DataFrame containing rolling window features
    """
    df = pd.DataFrame(index=time_series.index)
    
    # Map function names to actual functions
    func_map = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max,
        'median': np.median,
        'sum': np.sum,
        'var': np.var,
        'skew': stats.skew,
        'kurt': stats.kurtosis
    }
    
    # Filter functions
    functions = [f for f in functions if f in func_map]
    
    # Create rolling window features
    for window in windows:
        if window < len(time_series):  # Only create windows that make sense for the data
            for func_name in functions:
                func = func_map[func_name]
                df[f'rolling_{window}_{func_name}'] = time_series.rolling(window=window, min_periods=1).apply(func, raw=True)
    
    return df

def create_ewm_features(time_series, spans=[3, 6, 12]):
    """
    Create exponentially weighted moving average features
    
    Parameters:
    -----------
    time_series : pandas Series
        Time series data
    spans : list
        List of span values for EWM calculations
        
    Returns:
    --------
    ewm_features : pandas DataFrame
        DataFrame containing EWM features
    """
    df = pd.DataFrame(index=time_series.index)
    
    # Create EWM features
    for span in spans:
        df[f'ewm_{span}_mean'] = time_series.ewm(span=span, adjust=False).mean()
        df[f'ewm_{span}_std'] = time_series.ewm(span=span, adjust=False).std()
    
    return df

def create_diff_features(time_series, periods=[1, 2, 12]):
    """
    Create differenced features from time series
    
    Parameters:
    -----------
    time_series : pandas Series
        Time series data
    periods : list
        List of periods for differencing
        
    Returns:
    --------"
    "    time_features : pandas DataFrame
        DataFrame containing time-based features
    """
    # Extract datetime components
    dates = time_series.index
    
    df = pd.DataFrame(index=dates)
    
    # Basic date components
    df['Month'] = dates.month
    df['Quarter'] = dates.quarter
    df['Year'] = dates.year
    df['DayOfYear'] = dates.dayofyear
    
    # Cyclic features for month and quarter (to handle periodicity)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Quarter_sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
    df['Quarter_cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)
    
    # Year trend
    min_year = df['Year'].min()
    df['YearTrend'] = df['Year'] - min_year + (df['Month'] - 1) / 12
    
    # If data frequency allows, add more granular features
    if len(time_series) >= 24:  # Enough data points to be meaningful
        # Is beginning/end of quarter
        df['IsBeginQuarter'] = (df['Month'] % 3 == 1).astype(int)
        df['IsEndQuarter'] = (df['Month'] % 3 == 0).astype(int)
        
        # Is beginning/end of year
        df['IsBeginYear'] = (df['Month'] == 1).astype(int)
        df['IsEndYear'] = (df['Month'] == 12).astype(int)
    
    return df
