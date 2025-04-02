import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# Statistical models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import prophet

# Machine learning models
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor


def train_sarima_model(time_series, params):
    """
    Train a SARIMA model on time series data
    
    Parameters:
    -----------
    time_series : pandas Series
        Time series data
    params : dict
        Dictionary of model parameters
        - order: (p, d, q) tuple
        - seasonal_order: (P, D, Q, s) tuple
        
    Returns:
    --------
    model : SARIMAX model object
        Untrained model object
    model_fit : SARIMAXResults object
        Trained model object
    """
    # Get parameters, with defaults if not provided
    order = params.get("order", (1, 1, 1))
    seasonal_order = params.get("seasonal_order", (1, 1, 1, 12))
    
    # Create model
    model = SARIMAX(
        time_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    # Fit model
    model_fit = model.fit(disp=False)
    
    return model, model_fit

def train_prophet_model(time_series, params):
    """
    Train a Prophet model on time series data
    
    Parameters:
    -----------
    time_series : pandas Series
        Time series data
    params : dict
        Dictionary of model parameters
        - yearly_seasonality: bool
        - weekly_seasonality: bool
        - daily_seasonality: bool
        - growth: str ('linear' or 'logistic')
        - seasonality_mode: str ('additive' or 'multiplicative')
        
    Returns:
    --------
    model : Prophet model object
        Untrained model object
    model_fit : Prophet model object
        Trained model object
    """
    # Get parameters
    yearly_seasonality = params.get("yearly_seasonality", True)
    weekly_seasonality = params.get("weekly_seasonality", True)
    daily_seasonality = params.get("daily_seasonality", True)
    growth = params.get("growth", "linear")
    seasonality_mode = params.get("seasonality_mode", "additive")
    
    # Create model
    model = prophet.Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        growth=growth,
        seasonality_mode=seasonality_mode
    )
    
    # Prepare data for Prophet
    df = pd.DataFrame({
        'ds': time_series.index,
        'y': time_series.values
    })
    
    # Fit model
    model.fit(df)
    
    return model, model

def train_xgboost_model(time_series, features_df, params):
    """
    Train an XGBoost model on time series data with features
    
    Parameters:
    -----------
    time_series : pandas Series
        Time series data (target variable)
    features_df : pandas DataFrame
        Feature matrix
    params : dict
        Dictionary of model parameters
        - test_size: float (0-1)
        - max_depth: int
        - learning_rate: float
        - n_estimators: int
        - cv_folds: int
        
    Returns:
    --------
    model : XGBRegressor object
        Untrained model object
    model_fit : XGBRegressor object
        Trained model object
    """
    # Get parameters
    test_size = params.get("test_size", 0.2)
    max_depth = params.get("max_depth", 6)
    learning_rate = params.get("learning_rate", 0.1)
    n_estimators = params.get("n_estimators", 100)
    cv_folds = params.get("cv_folds", 5)
    
    # Prepare data
    X = features_df.copy()
    y = time_series.copy()
    
    # Drop NaN values (from lagged features, etc.)
    valid_indices = ~(X.isna().any(axis=1) | y.isna())
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    
    # Create train/test split
    if test_size > 0 and len(X) > 5:  # Only split if we have enough data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # No shuffling for time series
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None
    
    # Create model
    model = xgb.XGBRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )
    
    # Fit model
    if cv_folds > 1 and len(X_train) >= cv_folds + 1:
        # Cross-validation for time series
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Collect trained models from each fold
        models = []
        
        for train_idx, val_idx in tscv.split(X_train):
            fold_X_train, fold_X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            fold_y_train, fold_y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Train model on this fold
            fold_model = xgb.XGBRegressor(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                objective='reg:squarederror',
                n_jobs=-1,
                random_state=42
            )
            
            fold_model.fit(fold_X_train, fold_y_train,
                          eval_set=[(fold_X_val, fold_y_val)],
                          early_stopping_rounds=10,
                          verbose=False)
            
            models.append(fold_model)
        
        # Use the best model (last fold)
        model_fit = models[-1]
    else:
        # Simple fitting without CV
        eval_set = [(X_train, y_train)]
        if X_test is not None and y_test is not None:
            eval_set.append((X_test, y_test))
        
        model.fit(X_train, y_train,
                 eval_set=eval_set,
                 early_stopping_rounds=10 if len(eval_set) > 1 else None,
                 verbose=False)
        
        model_fit = model
    
    return model, model_fit

def train_lstm_model(time_series, features_df, params):
    """
    Train an LSTM model on time series data with features
    
    Parameters:
    -----------
    time_series : pandas Series
        Time series data (target variable)
    features_df : pandas DataFrame
        Feature matrix
    params : dict
        Dictionary of model parameters
        - test_size: float (0-1)
        - units: int
        - dropout: float
        - epochs: int
        
    Returns:
    --------
    model : Keras Model
        Untrained model object
    model_fit : Keras History
        Training history and model
    """
    # Get parameters
    test_size = params.get("test_size", 0.2)
    units = params.get("units", 64)
    dropout = params.get("dropout", 0.2)
    epochs = params.get("epochs", 50)
    
    # Prepare data
    X = features_df.copy()
    y = time_series.copy()
    
    # Drop NaN values (from lagged features, etc.)
    valid_indices = ~(X.isna().any(axis=1) | y.isna())
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    
    # Scale features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Scale target
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    # Convert to 3D shape expected by LSTM
    X_3d = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    # Create train/test split
    if test_size > 0 and len(X) > 5:  # Only split if we have enough data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X_3d[:split_idx], X_3d[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    else:
        X_train, y_train = X_3d, y_scaled
        X_test, y_test = None, None
    
    # Create model
    model = Sequential([
        LSTM(units, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(dropout),
        Dense(1)
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    # Fit model
    if X_test is not None and y_test is not None:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
    
    # Create a wrapper class to store model and scalers for later use
    class LSTMWrapper:
        def __init__(self, model, history, scaler_X, scaler_y):
            self.model = model
            self.history = history
            self.scaler_X = scaler_X
            self.scaler_y = scaler_y
        
        def predict(self, X):
            # Scale input
            X_scaled = self.scaler_X.transform(X)
            
            # Reshape for LSTM
            X_3d = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            
            # Make prediction
            y_pred_scaled = self.model.predict(X_3d)
            
            # Inverse transform
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
            
            return y_pred
    
    model_fit = LSTMWrapper(model, history, scaler_X, scaler_y)
    
    return model, model_fit

def evaluate_model(time_series, model_fit, model_type):
    """
    Evaluate a trained model on historical data
    
    Parameters:
    -----------
    time_series : pandas Series
        Historical time series data
    model_fit : trained model object
        Trained model to evaluate
    model_type : str
        Type of model ('SARIMA', 'Prophet', 'XGBoost', 'LSTM', 'Ensemble')
        
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Get predictions based on model type
    if model_type == "SARIMA":
        # In-sample predictions
        predictions = model_fit.fittedvalues
        
        # Align with actual values
        actuals = time_series.iloc[len(time_series) - len(predictions):]
    
    elif model_type == "Prophet":
        # Create DataFrame with dates for prediction
        df = pd.DataFrame({'ds': time_series.index})
        
        # Make predictions
        forecast = model_fit.predict(df)
        
        # Extract predictions
        predictions = forecast['yhat'].values
        
        # Actuals
        actuals = time_series.values
    
    elif model_type in ["XGBoost", "RandomForest"]:
        # If model was trained on features, we need those features
        # Here we assume features were generated from the time series
        
        # Dummy implementation - in practice you would need the features
        # Generate them the same way as during training
        actuals = time_series
        predictions = np.zeros(len(actuals))  # Placeholder
    
    elif model_type == "LSTM":
        # Similar to XGBoost, need to generate features
        
        # Dummy implementation
        actuals = time_series
        predictions = np.zeros(len(actuals))  # Placeholder
    
    elif model_type == "Ensemble":
        # For ensemble, combine predictions from multiple models
        
        # Dummy implementation
        actuals = time_series
        predictions = np.zeros(len(actuals))  # Placeholder
    
    else:
        # Fallback for unknown model types
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'r2': np.nan
        }
    
    # Ensure actuals and predictions are aligned and have the same length
    min_len = min(len(actuals), len(predictions))
    actuals = actuals[-min_len:]
    predictions = predictions[-min_len:]
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    # Calculate MAPE, handling zeros
    if np.any(actuals == 0):
        # Add small epsilon to avoid division by zero
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-10))) * 100
    else:
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    # Calculate R-squared
    r2 = r2_score(actuals, predictions)
    
    # Return metrics
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }
    
    return metrics

def generate_forecast(model_fit, forecast_periods, freq='MS', model_type='SARIMA', features=None):
    """
    Generate forecasts from a trained model
    
    Parameters:
    -----------
    model_fit : trained model object
        Trained model to use for forecasting
    forecast_periods : int
        Number of periods to forecast
    freq : str
        Frequency of the forecast (pandas frequency string)
    model_type : str
        Type of model ('SARIMA', 'Prophet', 'XGBoost', 'LSTM', 'Ensemble')
    features : pandas DataFrame, optional
        Feature matrix for machine learning models
        
    Returns:
    --------
    forecast : numpy array
        Array of forecasted values
    conf_int : numpy array
        Array of confidence intervals (lower, upper)
    """
    if model_type == "SARIMA":
        # Generate forecast
        forecast_result = model_fit.get_forecast(steps=forecast_periods)
        forecast = forecast_result.predicted_mean
        
        # Get confidence intervals
        conf_int = forecast_result.conf_int(alpha=0.05)
        conf_int = conf_int.values
    
    elif model_type == "Prophet":
        # Create future DataFrame
        future = model_fit.make_future_dataframe(periods=forecast_periods, freq=freq)
        
        # Make prediction
        forecast_df = model_fit.predict(future)
        
        # Extract forecast and confidence intervals
        forecast = forecast_df['yhat'].iloc[-forecast_periods:].values
        
        # Extract confidence intervals
        lower = forecast_df['yhat_lower'].iloc[-forecast_periods:].values
        upper = forecast_df['yhat_upper'].iloc[-forecast_periods:].values
        
        conf_int = np.column_stack((lower, upper))
    
    elif model_type == "XGBoost":
        # Need to generate features for future periods
        if features is not None:
            # Extremely simplified example - in practice, you would need to
            # generate features for future time periods
            
            # Dummy forecast
            forecast = np.ones(forecast_periods) * features.iloc[-1].mean()
            
            # Dummy confidence intervals
            std_dev = np.std(model_fit.predict(features)) if len(features) > 0 else 1.0
            lower = forecast - 1.96 * std_dev
            upper = forecast + 1.96 * std_dev
            
            conf_int = np.column_stack((lower, upper))
        else:
            # Fallback if no features provided
            forecast = np.ones(forecast_periods)
            conf_int = np.column_stack((forecast * 0.9, forecast * 1.1))
    
    elif model_type == "LSTM":
        # Similar to XGBoost, need to generate features for future periods
        if features is not None:
            # Extremely simplified example
            
            # Dummy forecast
            forecast = np.ones(forecast_periods) * features.iloc[-1].mean()
            
            # Dummy confidence intervals
            std_dev = 1.0  # Would calculate from model history in practice
            lower = forecast - 1.96 * std_dev
            upper = forecast + 1.96 * std_dev
            
            conf_int = np.column_stack((lower, upper))
        else:
            # Fallback if no features provided
            forecast = np.ones(forecast_periods)
            conf_int = np.column_stack((forecast * 0.9, forecast * 1.1))
    
    elif model_type == "Ensemble":
        # For ensemble, combine forecasts from multiple models
        
        # Dummy implementation
        forecast = np.ones(forecast_periods)
        conf_int = np.column_stack((forecast * 0.9, forecast * 1.1))
    
    else:
        # Fallback for unknown model types
        forecast = np.ones(forecast_periods)
        conf_int = np.column_stack((forecast * 0.9, forecast * 1.1))
    
    return forecast, conf_int