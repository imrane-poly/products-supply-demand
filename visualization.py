import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

def plot_time_series(time_series, title="Time Series", xlabel="Date", ylabel="Value", figsize=(10, 6), color='blue'):
    """
    Plot a time series using Plotly
    
    Parameters:
    -----------
    time_series : pandas Series
        Time series data with datetime index
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    color : str
        Line color
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object
    """
    fig = px.line(
        x=time_series.index,
        y=time_series.values,
        labels={'x': xlabel, 'y': ylabel},
        title=title
    )
    
    fig.update_traces(line=dict(color=color))
    
    fig.update_layout(
        height=figsize[1] * 80,
        width=figsize[0] * 80,
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=xlabel,
        yaxis_title=ylabel
    )
    
    return fig

def plot_multiple_time_series(series_dict, title="Multiple Time Series", xlabel="Date", ylabel="Value", figsize=(10, 6)):
    """
    Plot multiple time series in the same chart using Plotly
    
    Parameters:
    -----------
    series_dict : dict
        Dictionary of time series with format {name: series}
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    for name, series in series_dict.items():
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode='lines',
                name=name
            )
        )
    
    fig.update_layout(
        height=figsize[1] * 80,
        width=figsize[0] * 80,
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_forecast(historical_data, forecast_df, title="Forecast", xlabel="Date", ylabel="Value", figsize=(10, 6)):
    """
    Plot historical data and forecast with confidence intervals
    
    Parameters:
    -----------
    historical_data : pandas Series
        Historical time series data
    forecast_df : pandas DataFrame
        DataFrame containing forecast and confidence intervals
        Must have columns: ['Date', 'Forecast', 'Lower_CI', 'Upper_CI']
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data.values,
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        )
    )
    
    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        )
    )
    
    # Add confidence intervals
    fig.add_trace(
        go.Scatter(
            x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
            y=forecast_df['Upper_CI'].tolist() + forecast_df['Lower_CI'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(231,107,243,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval'
        )
    )
    
    fig.update_layout(
        height=figsize[1] * 80,
        width=figsize[0] * 80,
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_feature_importance(feature_importance, top_n=10, title="Feature Importance", figsize=(10, 6)):
    """
    Plot feature importance using Plotly
    
    Parameters:
    -----------
    feature_importance : pandas Series
        Series with feature names as index and importance values
    top_n : int
        Number of top features to display
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object
    """
    # Get top N features
    top_features = feature_importance.nlargest(top_n)
    
    # Create figure
    fig = px.bar(
        x=top_features.values,
        y=top_features.index,
        orientation='h',
        labels={'x': 'Importance', 'y': 'Feature'},
        title=title,
        color=top_features.values,
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(
        height=figsize[1] * 80,
        width=figsize[0] * 80,
        yaxis={'categoryorder': 'total ascending'},
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def plot_confusion_heatmap(data, x_col, y_col, value_col, title="Heatmap", 
                          figsize=(10, 8), colorscale="Viridis"):
    """
    Create a heatmap/confusion matrix using Plotly
    
    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing the data
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    value_col : str
        Column name for values
    title : str
        Plot title
    figsize : tuple
        Figure size
    colorscale : str
        Plotly colorscale name
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object
    """
    # Pivot the data
    pivot_data = data.pivot_table(
        index=y_col,
        columns=x_col,
        values=value_col,
        aggfunc='mean'
    ).fillna(0)
    
    fig = px.imshow(
        pivot_data,
        labels=dict(x=x_col, y=y_col, color=value_col),
        x=pivot_data.columns,
        y=pivot_data.index,
        title=title,
        color_continuous_scale=colorscale
    )
    
    fig.update_layout(
        height=figsize[1] * 80,
        width=figsize[0] * 80,
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def plot_correlation_matrix(data, columns=None, title="Correlation Matrix", figsize=(10, 8)):
    """
    Create a correlation matrix heatmap using Plotly
    
    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing the data
    columns : list or None
        List of columns to include in the correlation matrix
        If None, all numeric columns are used
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object
    """
    # Select columns if specified
    if columns is not None:
        data = data[columns]
    else:
        # Use all numeric columns
        data = data.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        labels=dict(x="Features", y="Features", color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.index,
        title=title,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )
    
    fig.update_layout(
        height=figsize[1] * 80,
        width=figsize[0] * 80,
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def plot_seasonal_decomposition(decomposition, title="Seasonal Decomposition", figsize=(12, 10)):
    """
    Plot seasonal decomposition components using Plotly
    
    Parameters:
    -----------
    decomposition : statsmodels DecomposeResult
        Result of seasonal_decompose
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        vertical_spacing=0.1
    )
    
    # Add observed data
    fig.add_trace(
        go.Scatter(
            x=decomposition.observed.index,
            y=decomposition.observed.values,
            mode="lines",
            name="Observed"
        ),
        row=1,
        col=1
    )
    
    # Add trend
    fig.add_trace(
        go.Scatter(
            x=decomposition.trend.index,
            y=decomposition.trend.values,
            mode="lines",
            name="Trend"
        ),
        row=2,
        col=1
    )
    
    # Add seasonal component
    fig.add_trace(
        go.Scatter(
            x=decomposition.seasonal.index,
            y=decomposition.seasonal.values,
            mode="lines",
            name="Seasonal"
        ),
        row=3,
        col=1
    )
    
    # Add residual
    fig.add_trace(
        go.Scatter(
            x=decomposition.resid.index,
            y=decomposition.resid.values,
            mode="lines",
            name="Residual"
        ),
        row=4,
        col=1
    )
    
    fig.update_layout(
        height=figsize[1] * 80,
        width=figsize[0] * 80,
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=False
    )
    
    return fig

def plot_shap_summary(shap_values, features, top_n=10, title="SHAP Summary Plot", figsize=(10, 6)):
    """
    Create a SHAP summary plot using Plotly
    
    Parameters:
    -----------
    shap_values : numpy array
        SHAP values from shap.Explainer
    features : pandas DataFrame
        Feature values used for SHAP calculation
    top_n : int
        Number of top features to display
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object
    """
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=features.columns)
    
    # Get top N features
    top_features_idx = feature_importance.nlargest(top_n).index
    
    # Filter SHAP values and features for top features only
    top_shap_values = shap_values[:, [features.columns.get_loc(col) for col in top_features_idx]]
    top_features = features[top_features_idx]
    
    # Create a DataFrame for visualization
    shap_df = pd.DataFrame()
    for i, feature in enumerate(top_features_idx):
        feature_shap = top_shap_values[:, i]
        feature_value = top_features[feature].values
        
        # Normalize feature values for color scaling
        feature_value_norm = (feature_value - feature_value.min()) / (feature_value.max() - feature_value.min() + 1e-8)
        
        temp_df = pd.DataFrame({
            'Feature': feature,
            'SHAP Value': feature_shap,
            'Feature Value': feature_value,
            'Feature Value Normalized': feature_value_norm,
            'Position': i
        })
        
        shap_df = pd.concat([shap_df, temp_df])
    
    # Create figure
    fig = px.strip(
        shap_df,
        y='Feature',
        x='SHAP Value',
        color='Feature Value Normalized',
        color_continuous_scale='RdBu_r',
        title=title,
        stripmode='overlay',
        orientation='h'
    )
    
    fig.update_layout(
        height=figsize[1] * 80,
        width=figsize[0] * 80,
        yaxis={'categoryorder': 'array', 'categoryarray': top_features_idx[::-1]},
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def plot_shap_dependence(shap_values, features, feature_idx, interaction_idx=None, 
                        title=None, figsize=(10, 6)):
    """
    Create a SHAP dependence plot using Plotly
    
    Parameters:
    -----------
    shap_values : numpy array
        SHAP values from shap.Explainer
    features : pandas DataFrame
        Feature values used for SHAP calculation
    feature_idx : int or str
        Index or name of the feature to plot
    interaction_idx : int or str, optional
        Index or name of the interaction feature
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object
    """
    # Convert feature_idx to column name if it's an integer
    if isinstance(feature_idx, int):
        feature_name = features.columns[feature_idx]
        feature_idx = feature_idx
    else:
        feature_name = feature_idx
        feature_idx = features.columns.get_loc(feature_idx)
    
    # Set title if not provided
    if title is None:
        title = f"SHAP Dependence Plot for {feature_name}"
    
    # Get feature values and SHAP values
    feature_values = features.iloc[:, feature_idx].values
    feature_shap_values = shap_values[:, feature_idx]
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Feature Value': feature_values,
        'SHAP Value': feature_shap_values
    })
    
    # Handle interaction feature
    if interaction_idx is not None:
        # Convert interaction_idx to column name if it's an integer
        if isinstance(interaction_idx, int):
            interaction_name = features.columns[interaction_idx]
            interaction_idx = interaction_idx
        else:
            interaction_name = interaction_idx
            interaction_idx = features.columns.get_loc(interaction_idx)
        
        # Get interaction feature values
        interaction_values = features.iloc[:, interaction_idx].values
        
        # Add to DataFrame
        plot_df['Interaction Value'] = interaction_values
        
        # Normalize interaction values for color scaling
        plot_df['Interaction Value Normalized'] = (plot_df['Interaction Value'] - plot_df['Interaction Value'].min()) / \
                                                (plot_df['Interaction Value'].max() - plot_df['Interaction Value'].min() + 1e-8)
        
        # Create scatter plot with interaction color
        fig = px.scatter(
            plot_df,
            x='Feature Value',
            y='SHAP Value',
            color='Interaction Value',
            color_continuous_scale='RdBu_r',
            opacity=0.7,
            title=title
        )
        
        # Update color axis title
        fig.update_coloraxes(colorbar_title=interaction_name)
    else:
        # Create scatter plot without interaction
        fig = px.scatter(
            plot_df,
            x='Feature Value',
            y='SHAP Value',
            opacity=0.7,
            title=title
        )
    
    # Add trend line
    fig.add_trace(
        go.Scatter(
            x=plot_df['Feature Value'],
            y=np.poly1d(np.polyfit(plot_df['Feature Value'], plot_df['SHAP Value'], 1))(plot_df['Feature Value']),
            mode='lines',
            name='Trend',
            line=dict(color='black')
        )
    )
    
    # Add horizontal zero line
    fig.add_shape(
        type="line",
        x0=min(plot_df['Feature Value']),
        y0=0,
        x1=max(plot_df['Feature Value']),
        y1=0,
        line=dict(color="red", dash="dash")
    )
    
    fig.update_layout(
        height=figsize[1] * 80,
        width=figsize[0] * 80,
        xaxis_title=feature_name,
        yaxis_title="SHAP Value",
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def plot_cross_validation_results(cv_results, title="Cross-Validation Results", figsize=(10, 6)):
    """
    Plot cross-validation results using Plotly
    
    Parameters:
    -----------
    cv_results : pandas DataFrame
        DataFrame containing cross-validation results
        Must have columns for fold, train_score, and test_score
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add train scores
    fig.add_trace(
        go.Scatter(
            x=cv_results['fold'],
            y=cv_results['train_score'],
            mode='lines+markers',
            name='Train Score',
            line=dict(color='blue')
        )
    )
    
    # Add test scores
    fig.add_trace(
        go.Scatter(
            x=cv_results['fold'],
            y=cv_results['test_score'],
            mode='lines+markers',
            name='Test Score',
            line=dict(color='red')
        )
    )
    
    # Add horizontal line for mean test score
    mean_test_score = cv_results['test_score'].mean()
    fig.add_shape(
        type="line",
        x0=cv_results['fold'].min(),
        y0=mean_test_score,
        x1=cv_results['fold'].max(),
        y1=mean_test_score,
        line=dict(color="red", dash="dash")
    )
    
    # Add text annotation for mean test score
    fig.add_annotation(
        x=cv_results['fold'].max(),
        y=mean_test_score,
        text=f"Mean Test Score: {mean_test_score:.4f}",
        showarrow=False,
        xshift=10,
        align="left"
    )
    
    fig.update_layout(
        height=figsize[1] * 80,
        width=figsize[0] * 80,
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Fold",
        yaxis_title="Score",
        xaxis=dict(tickmode='linear')
    )
    
    return fig

def plot_residuals(actual, predicted, title="Residual Analysis", figsize=(10, 12)):
    """
    Create a comprehensive residual analysis plot using Plotly
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object
    """
    # Calculate residuals
    residuals = np.array(actual) - np.array(predicted)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Residuals vs Predicted", "Residual Distribution", "Actual vs Predicted"),
        vertical_spacing=0.1
    )
    
    # Add residuals vs predicted
    fig.add_trace(
        go.Scatter(
            x=predicted,
            y=residuals,
            mode='markers',
            marker=dict(color='blue', opacity=0.6),
            name='Residuals'
        ),
        row=1,
        col=1
    )
    
    # Add horizontal zero line
    fig.add_shape(
        type="line",
        x0=min(predicted),
        y0=0,
        x1=max(predicted),
        y1=0,
        line=dict(color="red", dash="dash"),
        row=1,
        col=1
    )
    
    # Add residual histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            marker=dict(color='blue'),
            name='Residual Distribution'
        ),
        row=2,
        col=1
    )
    
    # Add actual vs predicted scatter plot
    fig.add_trace(
        go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            marker=dict(color='blue', opacity=0.6),
            name='Actual vs Predicted'
        ),
        row=3,
        col=1
    )
    
    # Add diagonal line (perfect predictions)
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        ),
        row=3,
        col=1
    )
    
    fig.update_layout(
        height=figsize[1] * 80,
        width=figsize[0] * 80,
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=False
    )
    
    # Update x and y axis labels
    fig.update_xaxes(title_text="Predicted Value", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=1, col=1)
    
    fig.update_xaxes(title_text="Residual", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    fig.update_xaxes(title_text="Actual Value", row=3, col=1)
    fig.update_yaxes(title_text="Predicted Value", row=3, col=1)
    
    return fig

def create_metrics_dashboard(metrics_dict, title="Model Performance Metrics"):
    """
    Create a metrics dashboard using Streamlit columns
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of metrics with format {name: value}
    title : str
        Dashboard title
    """
    st.subheader(title)
    
    # Create columns for metrics
    metrics_per_row = 4
    num_metrics = len(metrics_dict)
    num_rows = (num_metrics + metrics_per_row - 1) // metrics_per_row
    
    for row in range(num_rows):
        cols = st.columns(metrics_per_row)
        
        for col_idx in range(metrics_per_row):
            metric_idx = row * metrics_per_row + col_idx
            
            if metric_idx < num_metrics:
                metric_name = list(metrics_dict.keys())[metric_idx]
                metric_value = metrics_dict[metric_name]
                
                # Format metric value based on type
                if isinstance(metric_value, float):
                    formatted_value = f"{metric_value:.4f}"
                else:
                    formatted_value = str(metric_value)
                
                cols[col_idx].metric(metric_name, formatted_value)

def plot_scatter_matrix(data, dimensions=None, title="Scatter Matrix", figsize=(10, 10)):
    """
    Create a scatter matrix using Plotly
    
    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing the data
    dimensions : list or None
        List of columns to include in the scatter matrix
        If None, all numeric columns are used
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object
    """
    # Select dimensions if not specified
    if dimensions is None:
        dimensions = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create scatter matrix
    fig = px.scatter_matrix(
        data,
        dimensions=dimensions,
        title=title,
        opacity=0.7
    )
    
    fig.update_layout(
        height=figsize[0] * 80,
        width=figsize[1] * 80,
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def plot_geo_choropleth(geo_data, locations_col, values_col, title="Geographic Distribution", 
                       color_scale="Viridis", figsize=(10, 8)):
    """
    Create a choropleth map using Plotly
    
    Parameters:
    -----------
    geo_data : pandas DataFrame
        DataFrame containing the geographic data
    locations_col : str
        Column name for location identifiers (e.g., country codes)
    values_col : str
        Column name for values to be plotted
    title : str
        Plot title
    color_scale : str
        Plotly colorscale name
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : plotly Figure
        Plotly figure object
    """
    fig = px.choropleth(
        geo_data,
        locations=locations_col,
        color=values_col,
        title=title,
        color_continuous_scale=color_scale,
        projection="natural earth"
    )
    
    fig.update_layout(
        height=figsize[1] * 80,
        width=figsize[0] * 80,
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig