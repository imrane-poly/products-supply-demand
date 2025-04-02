import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import shap
from sklearn.inspection import permutation_importance

def calculate_feature_importance(model, features_df, model_type):
    """
    Calculate feature importance for various model types
    
    Parameters:
    -----------
    model : trained model object
        The trained model for which to calculate feature importance
    features_df : pandas DataFrame
        DataFrame containing the features used in the model
    model_type : str
        Type of model ('XGBoost', 'LSTM', 'Ensemble')
        
    Returns:
    --------
    feature_importance : pandas Series
        Series with feature names as index and importance values
    """
    if features_df.empty:
        return pd.Series()
    
    if model_type == "XGBoost":
        # Direct feature importance from XGBoost
        importance = pd.Series(model.feature_importances_, index=features_df.columns)
        return importance.sort_values(ascending=False)
    
    elif model_type == "LSTM":
        # For LSTM, use permutation importance
        # This is a simplified version - in reality, you'd need the LSTM model and validation data
        # to properly calculate permutation importance
        
        # Dummy implementation for demonstration
        importance = pd.Series(np.random.random(size=len(features_df.columns)), index=features_df.columns)
        importance = importance / importance.sum()
        return importance.sort_values(ascending=False)
    
    elif model_type == "Ensemble":
        # For ensemble, calculate the average importance across models that provide it
        importances = []
        
        for model_name, model_obj in model.items():
            if model_name == "XGBoost":
                # Add XGBoost importance
                imp = pd.Series(model_obj.feature_importances_, index=features_df.columns)
                importances.append(imp)
            elif model_name == "RandomForest":
                # Add RandomForest importance
                imp = pd.Series(model_obj.feature_importances_, index=features_df.columns)
                importances.append(imp)
        
        if importances:
            # Calculate average importance
            avg_importance = pd.concat(importances, axis=1).mean(axis=1)
            return avg_importance.sort_values(ascending=False)
        else:
            # Fallback if no models provide direct importance
            return pd.Series(1/len(features_df.columns), index=features_df.columns)
    
    else:
        # Default to equal importance if model doesn't support it
        return pd.Series(1/len(features_df.columns), index=features_df.columns)

def generate_shap_values(model, features_df, model_type):
    """
    Generate SHAP values for model explainability
    
    Parameters:
    -----------
    model : trained model object
        The trained model for which to generate SHAP values
    features_df : pandas DataFrame
        DataFrame containing the features used in the model
    model_type : str
        Type of model ('XGBoost', 'LSTM', 'Ensemble')
        
    Returns:
    --------
    shap_values : numpy array
        SHAP values for each feature and instance
    """
    if features_df.empty:
        return np.array([])
    
    try:
        if model_type == "XGBoost":
            # Create explainer
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(features_df)
            
            return shap_values
        
        elif model_type == "LSTM":
            # For deep learning models like LSTM, use KernelExplainer
            # This is computationally expensive, so we use a subset of data
            
            # Sample a subset of data for SHAP calculation
            sample_size = min(100, len(features_df))
            sample_data = features_df.sample(sample_size, random_state=42)
            
            # Define a prediction function for the model
            def predict_func(X):
                # This would need to be adapted to your actual model
                # For example, reshape X for LSTM and convert to proper format
                # Then get predictions
                return np.array([0.5] * len(X))  # Dummy return
            
            # Create explainer
            explainer = shap.KernelExplainer(predict_func, sample_data)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(sample_data)
            
            return shap_values
        
        elif model_type == "Ensemble":
            # For ensemble, use the first model that supports SHAP
            if "XGBoost" in model:
                explainer = shap.TreeExplainer(model["XGBoost"])
                shap_values = explainer.shap_values(features_df)
                return shap_values
            else:
                # Fallback
                return np.zeros((len(features_df), len(features_df.columns)))
        
        else:
            # Default if model doesn't support SHAP
            return np.zeros((len(features_df), len(features_df.columns)))
    
    except Exception as e:
        st.warning(f"Could not calculate SHAP values: {e}")
        return np.zeros((len(features_df), len(features_df.columns)))

def generate_model_insights(model, historical_data, forecast_df, model_type):
    """
    Generate natural language insights based on model and forecast
    
    Parameters:
    -----------
    model : trained model object
        The trained model used for forecasting
    historical_data : pandas Series
        Historical time series data
    forecast_df : pandas DataFrame
        DataFrame containing the forecast results
    model_type : str
        Type of model used for forecasting
        
    Returns:
    --------
    insights : str
        Natural language insights about the model and forecast
    """
    # Calculate basic statistics
    hist_mean = historical_data.mean()
    hist_std = historical_data.std()
    forecast_mean = forecast_df["Forecast"].mean()
    forecast_std = forecast_df["Forecast"].std()
    
    # Calculate trends
    hist_trend = (historical_data.iloc[-1] - historical_data.iloc[0]) / len(historical_data)
    forecast_trend = (forecast_df["Forecast"].iloc[-1] - forecast_df["Forecast"].iloc[0]) / len(forecast_df)
    
    # Determine if trend is accelerating or decelerating
    trend_change = "accelerating" if abs(forecast_trend) > abs(hist_trend) else "decelerating"
    
    # Calculate volatility
    hist_volatility = hist_std / hist_mean
    forecast_volatility = forecast_std / forecast_mean
    
    # Compare forecast to historical trends
    if forecast_mean > hist_mean:
        level_change = f"increasing by {((forecast_mean/hist_mean)-1)*100:.1f}%"
    else:
        level_change = f"decreasing by {((hist_mean/forecast_mean)-1)*100:.1f}%"
    
    # Generate insights based on model type
    model_specific_insights = ""
    
    if model_type == "SARIMA":
        model_specific_insights = """
        The SARIMA model captures both trend and seasonal patterns in the data.
        
        - The forecast reflects the historical seasonality pattern
        - Trend components are projected forward with adjustments for recent changes
        - The model typically performs well for data with clear seasonal patterns
        """
    
    elif model_type == "Prophet":
        model_specific_insights = """
        The Prophet model decomposes the time series into trend, seasonal, and holiday components.
        
        - The forecast accounts for multiple seasonality patterns (e.g., yearly, weekly)
        - Trend changepoints are automatically detected and incorporated
        - The model is robust to missing data and outliers
        """
    
    elif model_type == "XGBoost":
        model_specific_insights = """
        The XGBoost model uses gradient boosting with decision trees to capture complex patterns.
        
        - The forecast is based on a wide range of feature interactions
        - The model prioritizes recent patterns while still accounting for historical trends
        - Feature importance should be examined to understand key drivers
        """
    
    elif model_type == "LSTM":
        model_specific_insights = """
        The LSTM neural network captures complex sequential patterns in the data.
        
        - The model's memory cells can retain information over extended periods
        - Complex non-linear relationships are captured in the forecast
        - The forecast may reflect subtle patterns not captured by simpler models
        """
    
    elif model_type == "Ensemble":
        model_specific_insights = """
        The ensemble approach combines multiple models to improve forecast accuracy.
        
        - The final forecast represents a consensus view across several modeling approaches
        - This generally provides more robust predictions than any single model
        - The ensemble helps mitigate the weaknesses of individual models
        """
    
    # Combine insights
    insights = f"""
    ## Model Insights
    
    The {model_type} model forecasts a period of {trend_change} {'growth' if forecast_trend > 0 else 'decline'}, 
    with average values {level_change} compared to historical data.
    
    ### Volatility Analysis
    
    - Historical volatility: {hist_volatility:.2%}
    - Forecast volatility: {forecast_volatility:.2%}
    - The forecast shows {'higher' if forecast_volatility > hist_volatility else 'lower'} volatility compared to historical patterns.
    
    ### Model-Specific Insights
    {model_specific_insights}
    
    ### Key Considerations
    
    - The confidence interval {'widens' if forecast_df['Upper_CI'].iloc[-1] - forecast_df['Lower_CI'].iloc[-1] > forecast_df['Upper_CI'].iloc[0] - forecast_df['Lower_CI'].iloc[0] else 'remains stable'} over time, indicating {'increasing' if forecast_df['Upper_CI'].iloc[-1] - forecast_df['Lower_CI'].iloc[-1] > forecast_df['Upper_CI'].iloc[0] - forecast_df['Lower_CI'].iloc[0] else 'consistent'} forecast uncertainty.
    - The model {'maintains' if abs(forecast_trend - hist_trend) < 0.1 * abs(hist_trend) else 'shifts from'} the historical trend pattern.
    - External factors not captured in the model may impact actual outcomes.
    """
    
    return insights

def plot_feature_importance(feature_importance, top_n=10):
    """
    Create a plotly visualization of feature importance
    
    Parameters:
    -----------
    feature_importance : pandas Series
        Series with feature names as index and importance values
    top_n : int
        Number of top features to display
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object with feature importance visualization
    """
    # Get top N features
    top_features = feature_importance.nlargest(top_n)
    
    # Create figure
    fig = px.bar(
        x=top_features.values,
        y=top_features.index,
        orientation='h',
        title=f"Top {top_n} Feature Importance",
        labels={'x': 'Importance', 'y': 'Feature'}
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400
    )
    
    return fig

def plot_shap_summary(shap_values, features_df, top_n=10):
    """
    Create a plotly visualization of SHAP values summary
    
    Parameters:
    -----------
    shap_values : numpy array
        SHAP values from shap.Explainer
    features_df : pandas DataFrame
        Feature values used for SHAP calculation
    top_n : int
        Number of top features to display
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object with SHAP summary visualization
    """
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=features_df.columns)
    
    # Get top N features
    top_features = feature_importance.nlargest(top_n)
    
    # Create figure
    fig = px.bar(
        x=top_features.values,
        y=top_features.index,
        orientation='h',
        title=f"Top {top_n} Features by SHAP Impact",
        labels={'x': 'Mean |SHAP Value|', 'y': 'Feature'}
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400
    )
    
    return fig

def create_shap_dependence_plot(shap_values, features_df, feature_idx):
    """
    Create a SHAP dependence plot for a specific feature
    
    Parameters:
    -----------
    shap_values : numpy array
        SHAP values from shap.Explainer
    features_df : pandas DataFrame
        Feature values used for SHAP calculation
    feature_idx : int
        Index of the feature to analyze
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object with SHAP dependence plot
    """
    feature_name = features_df.columns[feature_idx]
    feature_values = features_df.iloc[:, feature_idx].values
    
    # Create figure
    fig = px.scatter(
        x=feature_values,
        y=shap_values[:, feature_idx],
        title=f"SHAP Dependence Plot for {feature_name}",
        labels={'x': feature_name, 'y': 'SHAP Value'},
        opacity=0.7
    )
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=min(feature_values),
        y0=0,
        x1=max(feature_values),
        y1=0,
        line=dict(color="red", dash="dash")
    )
    
    # Add trend line
    fig.add_trace(
        go.Scatter(
            x=feature_values,
            y=np.poly1d(np.polyfit(feature_values, shap_values[:, feature_idx], 1))(feature_values),
            mode='lines',
            name='Trend',
            line=dict(color='black')
        )
    )
    
    fig.update_layout(height=400)
    
    return fig

def partial_dependence_plot(model, X, feature_idx, feature_name, grid_resolution=50):
    """
    Create a partial dependence plot for a specific feature
    
    Parameters:
    -----------
    model : trained model object
        The trained model for which to create the partial dependence plot
    X : pandas DataFrame
        Feature matrix
    feature_idx : int
        Index of the feature to analyze
    feature_name : str
        Name of the feature
    grid_resolution : int
        Number of points in the grid
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object with partial dependence plot
    """
    # Get feature values
    feature_values = X.iloc[:, feature_idx].values
    
    # Create grid
    grid = np.linspace(
        np.percentile(feature_values, 5),
        np.percentile(feature_values, 95),
        grid_resolution
    )
    
    # Calculate predictions for each grid point
    mean_predictions = []
    
    for val in grid:
        # Create a copy of the data
        X_copy = X.copy()
        
        # Set feature to the current grid value
        X_copy.iloc[:, feature_idx] = val
        
        # Get predictions
        preds = model.predict(X_copy)
        
        # Calculate mean prediction
        mean_predictions.append(np.mean(preds))
    
    # Create figure
    fig = px.line(
        x=grid,
        y=mean_predictions,
        title=f"Partial Dependence Plot for {feature_name}",
        labels={'x': feature_name, 'y': 'Average Prediction'}
    )
    
    # Add distribution of actual feature values as histogram
    fig2 = px.histogram(
        x=feature_values,
        opacity=0.3,
        nbins=30
    )
    
    # Combine the figures
    fig.add_trace(go.Histogram(
        x=feature_values,
        y=np.ones(len(feature_values)) * min(mean_predictions),
        nbinsx=30,
        opacity=0.3,
        name="Distribution",
        yaxis="y2"
    ))
    
    fig.update_layout(
        height=400,
        yaxis2=dict(
            title="Count",
            overlaying="y",
            side="right",
            range=[0, len(X) / 5]  # Adjust scale
        )
    )
    
    return fig

def analyze_forecast_decomposition(model, forecast_df, model_type):
    """
    Decompose forecast into trend, seasonal, and residual components
    
    Parameters:
    -----------
    model : trained model object
        The trained model used for forecasting
    forecast_df : pandas DataFrame
        DataFrame containing the forecast results
    model_type : str
        Type of model ('SARIMA', 'Prophet', etc.)
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object with decomposition visualization
    """
    # This function will depend heavily on the model type
    # Here's a simplified implementation for demonstration
    
    if model_type == "Prophet" and hasattr(model, "component_modes"):
        # Prophet has built-in decomposition
        components = model.component_modes
        
        # Create figure with subplots
        fig = make_subplots(rows=4, cols=1, 
                          subplot_titles=("Forecast", "Trend", "Seasonality", "Residuals"))
        
        # Add forecast
        fig.add_trace(
            go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines", name="Forecast"),
            row=1, col=1
        )
        
        # Add trend (dummy implementation)
        trend = np.linspace(forecast_df["Forecast"].iloc[0], 
                           forecast_df["Forecast"].iloc[-1], 
                           len(forecast_df))
        
        fig.add_trace(
            go.Scatter(x=forecast_df["Date"], y=trend, mode="lines", name="Trend"),
            row=2, col=1
        )
        
        # Add seasonality (dummy implementation)
        seasonality = np.sin(np.linspace(0, 4*np.pi, len(forecast_df))) * forecast_df["Forecast"].std() * 0.5
        
        fig.add_trace(
            go.Scatter(x=forecast_df["Date"], y=seasonality, mode="lines", name="Seasonality"),
            row=3, col=1
        )
        
        # Add residuals (dummy implementation)
        residuals = forecast_df["Forecast"].values - trend - seasonality
        
        fig.add_trace(
            go.Scatter(x=forecast_df["Date"], y=residuals, mode="lines", name="Residuals"),
            row=4, col=1
        )
        
        fig.update_layout(height=800, title_text="Forecast Decomposition")
        
        return fig
    
    else:
        # Generic implementation for other models
        # This is very simplified and would need to be adapted for each model type
        
        # Create figure with subplots
        fig = make_subplots(rows=2, cols=1, 
                          subplot_titles=("Forecast", "Trend"))
        
        # Add forecast
        fig.add_trace(
            go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines", name="Forecast"),
            row=1, col=1
        )
        
        # Add simple trend (moving average)
        window = min(5, len(forecast_df) // 2)
        if window > 0:
            trend = forecast_df["Forecast"].rolling(window=window, center=True).mean()
            
            fig.add_trace(
                go.Scatter(x=forecast_df["Date"], y=trend, mode="lines", name=f"{window}-Point MA"),
                row=2, col=1
            )
        
        fig.update_layout(height=500, title_text="Forecast Trend Analysis")
        
        return fig

def analyze_anomalies(historical_data, forecast_df, threshold=2.0):
    """
    Detect and visualize anomalies in historical data and forecast
    
    Parameters:
    -----------
    historical_data : pandas Series
        Historical time series data
    forecast_df : pandas DataFrame
        DataFrame containing the forecast results
    threshold : float
        Z-score threshold for anomaly detection
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object with anomaly visualization
    anomalies : dict
        Dictionary with information about detected anomalies
    """
    # Combine historical and forecast data
    combined_df = pd.DataFrame({
        "Date": list(historical_data.index) + list(forecast_df["Date"]),
        "Value": list(historical_data.values) + list(forecast_df["Forecast"]),
        "Type": ["Historical"] * len(historical_data) + ["Forecast"] * len(forecast_df)
    })
    
    # Calculate rolling mean and std for z-score calculation
    window = min(12, len(historical_data) // 2)
    if window < 2:
        window = 2
    
    # Calculate historical stats
    historical_mean = historical_data.rolling(window=window).mean()
    historical_std = historical_data.rolling(window=window).std()
    
    # Calculate historical z-scores
    historical_z = (historical_data - historical_mean) / historical_std
    
    # Identify historical anomalies
    historical_anomalies = historical_data[abs(historical_z) > threshold]
    
    # For forecast, use historical stats for the first few points, then rolling
    forecast_z = []
    for i, val in enumerate(forecast_df["Forecast"]):
        if i == 0:
            # Use last historical stats
            mean = historical_mean.iloc[-1]
            std = historical_std.iloc[-1]
        else:
            # Use rolling window of forecast values
            lookback = min(i, window - 1)
            if lookback > 0:
                mean = forecast_df["Forecast"].iloc[max(0, i-lookback):i].mean()
                std = forecast_df["Forecast"].iloc[max(0, i-lookback):i].std()
            else:
                mean = val
                std = historical_std.iloc[-1]  # Use historical std as fallback
        
        # Calculate z-score, handling potential div by zero
        if std > 0:
            z = (val - mean) / std
        else:
            z = 0
            
        forecast_z.append(z)
    
    # Identify forecast anomalies
    forecast_anomalies = forecast_df.loc[abs(np.array(forecast_z)) > threshold, "Forecast"]
    forecast_anomaly_dates = forecast_df.loc[abs(np.array(forecast_z)) > threshold, "Date"]
    
    # Create visualization
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data.values,
        mode="lines",
        name="Historical",
        line=dict(color="blue")
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast"],
        mode="lines",
        name="Forecast",
        line=dict(color="red", dash="dash")
    ))
    
    # Add historical anomalies
    if not historical_anomalies.empty:
        fig.add_trace(go.Scatter(
            x=historical_anomalies.index,
            y=historical_anomalies.values,
            mode="markers",
            name="Historical Anomalies",
            marker=dict(color="blue", size=10, symbol="circle-open", line=dict(width=2))
        ))
    
    # Add forecast anomalies
    if len(forecast_anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_anomaly_dates,
            y=forecast_anomalies.values,
            mode="markers",
            name="Forecast Anomalies",
            marker=dict(color="red", size=10, symbol="circle-open", line=dict(width=2))
        ))
    
    # Update layout
    fig.update_layout(
        title="Anomaly Detection",
        xaxis_title="Date",
        yaxis_title="Value",
        height=500
    )
    
    # Prepare anomaly information
    anomalies = {
        "historical_count": len(historical_anomalies),
        "forecast_count": len(forecast_anomalies),
        "historical_dates": historical_anomalies.index.tolist() if not historical_anomalies.empty else [],
        "forecast_dates": forecast_anomaly_dates.tolist() if len(forecast_anomalies) > 0 else [],
        "historical_values": historical_anomalies.values.tolist() if not historical_anomalies.empty else [],
        "forecast_values": forecast_anomalies.values.tolist() if len(forecast_anomalies) > 0 else []
    }
    
    return fig, anomalies