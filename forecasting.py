import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Import modeling functions
from modules.modeling import (
    train_sarima_model,
    train_prophet_model,
    train_xgboost_model,
    train_lstm_model,
    evaluate_model,
    generate_forecast
)

# Import feature engineering
from modules.feature_engineering import (
    create_time_features,
    create_lagged_features,
    create_rolling_features
)

# Import explainability tools
from modules.explainability import (
    calculate_feature_importance,
    generate_shap_values,
    generate_model_insights
)

def show(data):
    """
    Forecasting page to train models and generate predictions
    """
    st.title("Supply & Demand Forecasting")
    st.write("Build, train and evaluate forecasting models for supply and demand predictions.")
    
    # Check if data is available
    if data.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
        return
    
    # Sidebar for model configuration
    st.sidebar.markdown("## Model Configuration")
    
    # Target selection
    available_targets = [col for col in data.columns if col == "ObservedValue"]
    if not available_targets:
        st.error("Cannot find target variable in the data.")
        return
    
    target_variable = available_targets[0]
    
    # Group selection
    group_by_options = ["CountryName", "Product", "FlowBreakdown", "CountryName + Product", "CountryName + FlowBreakdown"]
    grouping = st.sidebar.selectbox("Group By", group_by_options, index=3)
    
    # Create the appropriate grouping
    if "+" in grouping:
        group_cols = [col.strip() for col in grouping.split("+")]
    else:
        group_cols = [grouping]
    
    # Create combined identifier if multiple grouping columns
    if len(group_cols) > 1:
        data["Group"] = data[group_cols].apply(lambda x: " | ".join(x.astype(str)), axis=1)
        unique_groups = sorted(data["Group"].unique())
        selected_group = st.sidebar.selectbox("Select Group", unique_groups)
        
        # Filter for selected group
        selected_data = data[data["Group"] == selected_group].copy()
    else:
        unique_groups = sorted(data[group_cols[0]].unique())
        selected_group = st.sidebar.selectbox("Select Group", unique_groups)
        
        # Filter for selected group
        selected_data = data[data[group_cols[0]] == selected_group].copy()
    
    # Time aggregation
    aggregation_options = ["Monthly", "Quarterly", "Yearly"]
    time_aggregation = st.sidebar.selectbox("Time Aggregation", aggregation_options, index=0)
    
    # Convert to appropriate time frequency
    selected_data["ReferenceDate"] = pd.to_datetime(selected_data["ReferenceDate"])
    
    if time_aggregation == "Monthly":
        freq = "MS"
        selected_data["Period"] = selected_data["ReferenceDate"].dt.to_period("M")
    elif time_aggregation == "Quarterly":
        freq = "QS"
        selected_data["Period"] = selected_data["ReferenceDate"].dt.to_period("Q")
    else:  # Yearly
        freq = "YS"
        selected_data["Period"] = selected_data["ReferenceDate"].dt.to_period("Y")
    
    # Aggregate data
    agg_data = selected_data.groupby("Period")[target_variable].mean().reset_index()
    agg_data["Date"] = agg_data["Period"].dt.to_timestamp()
    
    # Sort by date
    agg_data = agg_data.sort_values("Date")
    
    # Display data
    st.subheader("Historical Data")
    
    # Line chart of historical data
    fig = px.line(
        agg_data,
        x="Date",
        y=target_variable,
        title=f"{selected_group} - Historical {time_aggregation} Data",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model selection
    model_options = {
        "SARIMA": "Statistical time series model with seasonal components",
        "Prophet": "Facebook's forecasting model for time series",
        "XGBoost": "Gradient boosting machine learning model",
        "LSTM": "Deep learning model for complex patterns",
        "Ensemble": "Combination of multiple models"
    }
    
    selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()))
    
    # Model description
    st.sidebar.markdown(f"**Model Description**: {model_options[selected_model]}")
    
    # Forecast horizon
    forecast_periods = st.sidebar.slider("Forecast Horizon", min_value=1, max_value=24, value=12)
    
    # Feature engineering options
    st.sidebar.markdown("## Feature Engineering")
    
    use_time_features = st.sidebar.checkbox("Add Time Features", value=True)
    use_lagged_features = st.sidebar.checkbox("Add Lagged Features", value=True)
    use_rolling_features = st.sidebar.checkbox("Add Rolling Features", value=True)
    
    # Add advanced options based on model
    if selected_model == "SARIMA":
        st.sidebar.markdown("## SARIMA Parameters")
        p = st.sidebar.slider("p (AR order)", 0, 5, 2)
        d = st.sidebar.slider("d (Differencing)", 0, 2, 1)
        q = st.sidebar.slider("q (MA order)", 0, 5, 2)
        
        seasonal_p = st.sidebar.slider("Seasonal P", 0, 2, 1)
        seasonal_d = st.sidebar.slider("Seasonal D", 0, 1, 1)
        seasonal_q = st.sidebar.slider("Seasonal Q", 0, 2, 1)
        seasonal_m = {"Monthly": 12, "Quarterly": 4, "Yearly": 1}[time_aggregation]
        
        model_params = {
            "order": (p, d, q),
            "seasonal_order": (seasonal_p, seasonal_d, seasonal_q, seasonal_m)
        }
    
    elif selected_model == "Prophet":
        st.sidebar.markdown("## Prophet Parameters")
        yearly_seasonality = st.sidebar.checkbox("Yearly Seasonality", value=True)
        weekly_seasonality = st.sidebar.checkbox("Weekly Seasonality", value=False)
        daily_seasonality = st.sidebar.checkbox("Daily Seasonality", value=False)
        
        growth = st.sidebar.selectbox("Growth Model", ["linear", "logistic"], index=0)
        seasonality_mode = st.sidebar.selectbox("Seasonality Mode", ["additive", "multiplicative"], index=0)
        
        model_params = {
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "daily_seasonality": daily_seasonality,
            "growth": growth,
            "seasonality_mode": seasonality_mode
        }
    
    elif selected_model in ["XGBoost", "LSTM", "Ensemble"]:
        st.sidebar.markdown(f"## {selected_model} Parameters")
        
        # Common ML parameters
        test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20) / 100
        cv_folds = st.sidebar.slider("Cross-Validation Folds", 2, 10, 5)
        
        model_params = {
            "test_size": test_size,
            "cv_folds": cv_folds
        }
        
        if selected_model == "XGBoost":
            max_depth = st.sidebar.slider("Max Depth", 3, 10, 6)
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1)
            n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, 100)
            
            model_params.update({
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "n_estimators": n_estimators
            })
        
        elif selected_model == "LSTM":
            units = st.sidebar.slider("LSTM Units", 32, 256, 64)
            dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)
            epochs = st.sidebar.slider("Epochs", 10, 200, 50)
            
            model_params.update({
                "units": units,
                "dropout": dropout,
                "epochs": epochs
            })
        
        elif selected_model == "Ensemble":
            include_sarima = st.sidebar.checkbox("Include SARIMA", value=True)
            include_prophet = st.sidebar.checkbox("Include Prophet", value=True)
            include_xgboost = st.sidebar.checkbox("Include XGBoost", value=True)
            include_lstm = st.sidebar.checkbox("Include LSTM", value=False)
            
            model_params.update({
                "include_sarima": include_sarima,
                "include_prophet": include_prophet,
                "include_xgboost": include_xgboost,
                "include_lstm": include_lstm
            })
    
    # Train and generate forecast when button is pressed
    if st.button("Train Model & Generate Forecast"):
        with st.spinner(f"Training {selected_model} model and generating forecast..."):
            # Prepare time series data
            time_series_data = agg_data.set_index("Date")[target_variable]
            
            # Feature engineering
            features_df = pd.DataFrame(index=time_series_data.index)
            
            if use_time_features:
                time_features = create_time_features(time_series_data)
                features_df = pd.concat([features_df, time_features], axis=1)
            
            if use_lagged_features:
                lag_features = create_lagged_features(time_series_data, lags=[1, 2, 3, 6, 12])
                features_df = pd.concat([features_df, lag_features], axis=1)
            
            if use_rolling_features:
                rolling_features = create_rolling_features(time_series_data, windows=[3, 6, 12])
                features_df = pd.concat([features_df, rolling_features], axis=1)
            
            # Train the appropriate model
            if selected_model == "SARIMA":
                model, model_fit = train_sarima_model(time_series_data, model_params)
                
                # Generate forecast
                forecast_result, conf_int = generate_forecast(
                    model_fit, 
                    forecast_periods, 
                    freq=freq
                )
                
            elif selected_model == "Prophet":
                model, model_fit = train_prophet_model(time_series_data, model_params)
                
                # Generate forecast
                forecast_result, conf_int = generate_forecast(
                    model_fit, 
                    forecast_periods, 
                    freq=freq, 
                    model_type="prophet"
                )
                
            elif selected_model == "XGBoost":
                model, model_fit = train_xgboost_model(time_series_data, features_df, model_params)
                
                # Generate forecast
                forecast_result, conf_int = generate_forecast(
                    model_fit, 
                    forecast_periods, 
                    freq=freq, 
                    model_type="xgboost",
                    features=features_df
                )
                
            elif selected_model == "LSTM":
                model, model_fit = train_lstm_model(time_series_data, features_df, model_params)
                
                # Generate forecast
                forecast_result, conf_int = generate_forecast(
                    model_fit, 
                    forecast_periods, 
                    freq=freq, 
                    model_type="lstm",
                    features=features_df
                )
                
            elif selected_model == "Ensemble":
                # Train multiple models and combine forecasts
                models = {}
                
                if model_params.get("include_sarima"):
                    sarima_model, sarima_fit = train_sarima_model(time_series_data, {
                        "order": (2, 1, 2),
                        "seasonal_order": (1, 1, 1, {"Monthly": 12, "Quarterly": 4, "Yearly": 1}[time_aggregation])
                    })
                    models["SARIMA"] = sarima_fit
                
                if model_params.get("include_prophet"):
                    prophet_model, prophet_fit = train_prophet_model(time_series_data, {
                        "yearly_seasonality": True,
                        "weekly_seasonality": False,
                        "daily_seasonality": False
                    })
                    models["Prophet"] = prophet_fit
                
                if model_params.get("include_xgboost"):
                    xgb_model, xgb_fit = train_xgboost_model(time_series_data, features_df, {
                        "max_depth": 6,
                        "learning_rate": 0.1,
                        "n_estimators": 100
                    })
                    models["XGBoost"] = xgb_fit
                
                if model_params.get("include_lstm"):
                    lstm_model, lstm_fit = train_lstm_model(time_series_data, features_df, {
                        "units": 64,
                        "dropout": 0.2,
                        "epochs": 50
                    })
                    models["LSTM"] = lstm_fit
                
                # Generate ensemble forecast
                forecast_result, conf_int = generate_forecast(
                    models, 
                    forecast_periods, 
                    freq=freq, 
                    model_type="ensemble",
                    features=features_df
                )
            
            # Evaluate model
            metrics = evaluate_model(time_series_data, model_fit, selected_model)
            
            # Feature importance (if available)
            if selected_model in ["XGBoost", "LSTM", "Ensemble"]:
                feature_importance = calculate_feature_importance(model_fit, features_df, selected_model)
                shap_values = generate_shap_values(model_fit, features_df, selected_model)
            
            # Create forecast index
            forecast_index = pd.date_range(start=time_series_data.index[-1], periods=forecast_periods + 1, freq=freq)[1:]
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                "Date": forecast_index,
                "Forecast": forecast_result,
                "Lower_CI": conf_int[:, 0],
                "Upper_CI": conf_int[:, 1]
            })
            
            # Display forecast results
            st.subheader("Forecast Results")
            
            # Visualization combining historical and forecast
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=time_series_data.index,
                y=time_series_data.values,
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Forecast"],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast_df["Date"].tolist() + forecast_df["Date"].tolist()[::-1],
                y=forecast_df["Upper_CI"].tolist() + forecast_df["Lower_CI"].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(231,107,243,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{selected_group} - {selected_model} Forecast",
                xaxis_title="Date",
                yaxis_title="Value",
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            st.subheader("Model Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("RMSE", f"{metrics['rmse']:.4f}")
            
            with col2:
                st.metric("MAE", f"{metrics['mae']:.4f}")
            
            with col3:
                st.metric("MAPE", f"{metrics['mape']:.2f}%")
            
            with col4:
                st.metric("R²", f"{metrics['r2']:.4f}")
            
            # Display forecast table
            st.subheader("Forecast Values")
            st.dataframe(
                forecast_df.set_index("Date").style.format({
                    "Forecast": "{:.2f}",
                    "Lower_CI": "{:.2f}",
                    "Upper_CI": "{:.2f}"
                })
            )
            
            # Feature importance visualization (if available)
            if selected_model in ["XGBoost", "LSTM", "Ensemble"] and use_time_features or use_lagged_features or use_rolling_features:
                st.subheader("Feature Importance")
                
                # Feature importance plot
                fig = px.bar(
                    x=feature_importance.index,
                    y=feature_importance.values,
                    labels={'x': 'Feature', 'y': 'Importance'},
                    title="Feature Importance"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add SHAP values visualization
                st.subheader("SHAP Values")
                
                # For simplicity, we'll just show a placeholder here
                # In a real implementation, you would use shap library's visualizations
                st.write("SHAP value visualization would be displayed here.")
                
                # Add model insights
                insights = generate_model_insights(model_fit, time_series_data, forecast_df, selected_model)
                
                st.subheader("Model Insights")
                st.write(insights)
            
            # Forecast insights
            st.subheader("Forecast Insights")
            
            # Calculate insights
            last_historical = time_series_data.iloc[-1]
            avg_forecast = forecast_df["Forecast"].mean()
            end_forecast = forecast_df["Forecast"].iloc[-1]
            
            pct_change_avg = ((avg_forecast - last_historical) / last_historical) * 100
            pct_change_end = ((end_forecast - last_historical) / last_historical) * 100
            
            trend_direction = "increasing" if pct_change_end > 0 else "decreasing"
            trend_magnitude = "significantly" if abs(pct_change_end) > 10 else "moderately" if abs(pct_change_end) > 5 else "slightly"
            
            # Display insights
            st.markdown(f"""
            ### Key Takeaways
            
            - The forecast shows a {trend_magnitude} {trend_direction} trend over the next {forecast_periods} periods.
            - Average forecasted value: {avg_forecast:.2f} ({pct_change_avg:+.2f}% from last historical value)
            - Final forecasted value: {end_forecast:.2f} ({pct_change_end:+.2f}% from last historical value)
            - Forecast uncertainty: {"High" if (forecast_df["Upper_CI"] - forecast_df["Lower_CI"]).mean() / avg_forecast > 0.2 else "Moderate" if (forecast_df["Upper_CI"] - forecast_df["Lower_CI"]).mean() / avg_forecast > 0.1 else "Low"}
            
            ### Additional Insights
            
            - {"There is significant seasonality in the data pattern." if "Month" in features_df.columns and feature_importance.get("Month_sin", 0) + feature_importance.get("Month_cos", 0) > 0.1 else "No significant seasonality detected in the data pattern."}
            - {"Recent values have the strongest influence on the forecast." if "lag_1" in features_df.columns and feature_importance.get("lag_1", 0) > 0.1 else "The forecast is influenced by a balanced mix of recent and historical patterns."}
            - {"The confidence interval widens significantly toward the end of the forecast, indicating increasing uncertainty for longer-term predictions." if (forecast_df["Upper_CI"] - forecast_df["Lower_CI"]).iloc[-1] > 1.5 * (forecast_df["Upper_CI"] - forecast_df["Lower_CI"]).iloc[0] else "The forecast maintains relatively consistent confidence levels throughout the prediction horizon."}
            """)
            
            # Download options
            st.subheader("Download Results")
            
            csv = forecast_df.to_csv().encode('utf-8')
            st.download_button(
                label="Download Forecast Data",
                data=csv,
                file_name=f"{selected_group}_{selected_model}_forecast.csv",
                mime="text/csv",
            )