import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Import custom modules
from modules.visualization import (
    plot_time_series,
    plot_multiple_time_series,
    plot_confusion_heatmap
)
from modules.data_loader import (
    get_time_series,
    get_comparable_flows,
    aggregate_data
)
from modules.utils import (
    display_info_card,
    display_metric_row,
    parse_date_range,
    calculate_growth_rate,
    get_trend_direction
)
import config

def show(data):
    """
    Supply & Demand Explorer page to explore and visualize data
    
    Parameters:
    -----------
    data : pandas DataFrame
        Processed data
    """
    st.title("Supply & Demand Explorer")
    st.write("Explore supply and demand patterns across countries, products, and time periods.")
    
    if data.empty:
        st.error("No data available. Please check data loading and filtering.")
        return
    
    # Create sidebar filters for exploration
    st.sidebar.markdown("## Data Filters")
    
    # Get unique values for filters
    countries = sorted(data["CountryName"].unique())
    products = sorted(data["Product"].unique())
    flow_metrics = sorted(data["FlowBreakdown"].unique())
    
    # Country filter
    selected_country = st.sidebar.selectbox(
        "Country", 
        ["All Countries"] + countries,
        index=0 if "All Countries" in ["All Countries"] + countries else 0
    )
    
    # Product filter
    selected_product = st.sidebar.selectbox(
        "Product", 
        products,
        index=0 if config.DEFAULT_PRODUCT in products else 0
    )
    
    # Flow breakdown filter
    selected_flow = st.sidebar.selectbox(
        "Flow Metric", 
        flow_metrics,
        index=0 if config.DEFAULT_FLOW in flow_metrics else 0
    )
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=[data["ReferenceDate"].min().date(), data["ReferenceDate"].max().date()],
        min_value=data["ReferenceDate"].min().date(),
        max_value=data["ReferenceDate"].max().date()
    )
    
    # Parse date range
    start_date, end_date = parse_date_range(date_range)
    
    # Filter data based on selections
    filtered_data = data.copy()
    
    # Filter by date
    filtered_data = filtered_data[
        (filtered_data["ReferenceDate"].dt.date >= start_date) & 
        (filtered_data["ReferenceDate"].dt.date <= end_date)
    ]
    
    # Filter by product
    filtered_data = filtered_data[filtered_data["Product"] == selected_product]
    
    # Filter by flow
    filtered_data = filtered_data[filtered_data["FlowBreakdown"] == selected_flow]
    
    # Filter by country if not "All Countries"
    if selected_country != "All Countries":
        filtered_data = filtered_data[filtered_data["CountryName"] == selected_country]
    
    # Create tabs for different views
    tabs = st.tabs([
        "Overview", 
        "Time Series Analysis", 
        "Country Comparison", 
        "Flow Analysis", 
        "Data Table"
    ])
    
    # Tab 1: Overview
    with tabs[0]:
        st.subheader("Supply & Demand Overview")
        
        # Calculate aggregates
        if selected_country == "All Countries":
            # Get top countries by average value
            country_agg = filtered_data.groupby("CountryName")["ObservedValue"].mean().reset_index()
            country_agg = country_agg.sort_values("ObservedValue", ascending=False)
            top_countries = country_agg.head(5)["CountryName"].tolist()
            
            # Filter for top countries
            top_country_data = filtered_data[filtered_data["CountryName"].isin(top_countries)]
            
            # Create time series for each top country
            country_time_series = {}
            
            for country in top_countries:
                country_data = top_country_data[top_country_data["CountryName"] == country]
                ts = get_time_series(
                    country_data, 
                    country, 
                    selected_product, 
                    selected_flow, 
                    freq='M'
                )
                country_time_series[country] = ts
            
            # Plot multiple time series
            fig = plot_multiple_time_series(
                country_time_series,
                title=f"Top 5 Countries: {selected_product} {selected_flow}",
                xlabel="Date",
                ylabel="Value (KB/D)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate global summary
            latest_date = filtered_data["ReferenceDate"].max()
            previous_year_date = latest_date - pd.DateOffset(years=1)
            
            latest_data = filtered_data[filtered_data["ReferenceDate"] == latest_date]
            previous_year_data = filtered_data[filtered_data["ReferenceDate"] == previous_year_date]
            
            current_total = latest_data["ObservedValue"].sum()
            previous_year_total = previous_year_data["ObservedValue"].sum() if not previous_year_data.empty else 0
            
            yoy_change = ((current_total - previous_year_total) / previous_year_total * 100) if previous_year_total > 0 else 0
            
            # Display summary metrics
            st.subheader("Global Summary")
            
            metrics = {
                "Current Total": f"{current_total:,.0f} KB/D",
                "YoY Change": f"{yoy_change:+.1f}%",
                "Top Contributor": top_countries[0] if top_countries else "N/A",
                "Countries": len(country_agg)
            }
            
            display_metric_row(metrics)
            
            # Create world summary chart
            st.subheader("Global Distribution")
            
            # Prepare data for pie chart
            pie_data = latest_data.groupby("CountryName")["ObservedValue"].sum().reset_index()
            pie_data = pie_data.sort_values("ObservedValue", ascending=False)
            
            # Group smaller countries as "Others"
            top_n = 10
            if len(pie_data) > top_n:
                others_sum = pie_data.iloc[top_n:]["ObservedValue"].sum()
                pie_data = pd.concat([
                    pie_data.iloc[:top_n],
                    pd.DataFrame({"CountryName": ["Others"], "ObservedValue": [others_sum]})
                ]).reset_index(drop=True)
            
            # Create pie chart
            fig = px.pie(
                pie_data,
                values="ObservedValue",
                names="CountryName",
                title=f"Global {selected_product} {selected_flow} Distribution",
                hole=0.4
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Regional analysis
            st.subheader("Regional Analysis")
            
            # Define regions (simplified)
            regions = {
                "North America": ["United States", "Canada", "Mexico"],
                "Europe": config.COUNTRY_GROUPS["EUROPE"],
                "Asia Pacific": config.COUNTRY_GROUPS["ASIA_PACIFIC"],
                "Middle East": ["Saudi Arabia", "Iran", "Iraq", "Kuwait", "UAE", "Qatar", "Oman", "Bahrain", "Yemen"],
                "Africa": ["Algeria", "Angola", "Nigeria", "Libya", "Egypt", "South Africa"],
                "Latin America": ["Brazil", "Venezuela", "Colombia", "Argentina", "Ecuador", "Peru", "Chile"]
            }
            
            # Map countries to regions
            country_to_region = {}
            for region, countries_list in regions.items():
                for country in countries_list:
                    country_to_region[country] = region
            
            # Add region column to dataset
            region_data = latest_data.copy()
            region_data["Region"] = region_data["CountryName"].map(country_to_region)
            region_data["Region"].fillna("Other", inplace=True)
            
            # Aggregate by region
            region_agg = region_data.groupby("Region")["ObservedValue"].sum().reset_index()
            region_agg = region_agg.sort_values("ObservedValue", ascending=False)
            
            # Create bar chart
            fig = px.bar(
                region_agg,
                x="Region",
                y="ObservedValue",
                title=f"Regional {selected_product} {selected_flow}",
                color="ObservedValue",
                color_continuous_scale="Viridis",
                text_auto='.2s'
            )
            
            fig.update_layout(
                xaxis_title="Region",
                yaxis_title="Value (KB/D)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Single country analysis
            st.subheader(f"{selected_country} - {selected_product} {selected_flow}")
            
            # Get time series
            country_ts = get_time_series(
                filtered_data, 
                selected_country, 
                selected_product, 
                selected_flow, 
                freq='M'
            )
            
            if not country_ts.empty:
                # Plot time series
                fig = plot_time_series(
                    country_ts,
                    title=f"{selected_country} - {selected_product} {selected_flow}",
                    xlabel="Date",
                    ylabel="Value (KB/D)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate metrics
                current_value = country_ts.iloc[-1] if not country_ts.empty else 0
                previous_year_value = country_ts.iloc[-13] if len(country_ts) > 12 else 0
                
                yoy_change = ((current_value - previous_year_value) / previous_year_value * 100) if previous_year_value > 0 else 0
                growth_rate = calculate_growth_rate(country_ts)
                
                # Get trend direction
                trend = get_trend_direction(country_ts)
                trend_icon = "📈" if trend == "up" else "📉" if trend == "down" else "➡️"
                
                # Display summary metrics
                metrics = {
                    "Current Value": f"{current_value:,.0f} KB/D",
                    "YoY Change": f"{yoy_change:+.1f}%",
                    "Trend": f"{trend_icon} {trend.title()}",
                    "CAGR": f"{growth_rate:.1f}%" if growth_rate is not None else "N/A"
                }
                
                display_metric_row(metrics)
                
                # Compare with similar countries
                st.subheader("Comparison with Similar Countries")
                
                # Get similar countries based on data magnitude
                country_values = filtered_data.groupby("CountryName")["ObservedValue"].mean().reset_index()
                current_country_value = country_values[country_values["CountryName"] == selected_country]["ObservedValue"].iloc[0]
                
                # Find countries with similar magnitude (within 50% of the current country)
                similar_countries = country_values[
                    (country_values["ObservedValue"] >= current_country_value * 0.5) &
                    (country_values["ObservedValue"] <= current_country_value * 1.5) &
                    (country_values["CountryName"] != selected_country)
                ]
                
                similar_countries = similar_countries.sort_values("ObservedValue", ascending=False)
                similar_countries = similar_countries.head(4)["CountryName"].tolist()
                
                if similar_countries:
                    # Create time series for similar countries
                    similar_ts = {selected_country: country_ts}
                    
                    for country in similar_countries:
                        ts = get_time_series(
                            filtered_data, 
                            country, 
                            selected_product, 
                            selected_flow, 
                            freq='M'
                        )
                        similar_ts[country] = ts
                    
                    # Plot comparison
                    fig = plot_multiple_time_series(
                        similar_ts,
                        title=f"Comparison with Similar Countries",
                        xlabel="Date",
                        ylabel="Value (KB/D)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No similar countries found for comparison.")
                
                # Compare with different flows
                st.subheader("Flow Comparison")
                
                # Get comparable flows
                comparable_flows = get_comparable_flows(data, selected_country, selected_product, selected_flow)
                
                if len(comparable_flows) > 1:
                    # Create time series for each flow
                    flow_ts = {}
                    
                    for flow in comparable_flows[:5]:  # Limit to top 5
                        ts = get_time_series(
                            data, 
                            selected_country, 
                            selected_product, 
                            flow, 
                            freq='M'
                        )
                        flow_ts[flow] = ts
                    
                    # Plot comparison
                    fig = plot_multiple_time_series(
                        flow_ts,
                        title=f"{selected_country} - {selected_product} Flow Comparison",
                        xlabel="Date",
                        ylabel="Value (KB/D)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No comparable flows found for comparison.")
            else:
                st.warning(f"No data available for {selected_country} - {selected_product} {selected_flow}.")
    
    # Tab 2: Time Series Analysis
    with tabs[1]:
        st.subheader("Time Series Analysis")
        
        # Create multi-country selector if "All Countries"
        if selected_country == "All Countries":
            multi_countries = st.multiselect(
                "Select Countries to Compare",
                countries,
                default=top_countries[:3] if 'top_countries' in locals() else countries[:3]
            )
            
            if not multi_countries:
                st.warning("Please select at least one country.")
                return
            
            # Filter data for selected countries
            multi_country_data = filtered_data[filtered_data["CountryName"].isin(multi_countries)]
            
            # Create time series for each country
            country_ts_dict = {}
            
            for country in multi_countries:
                country_data = multi_country_data[multi_country_data["CountryName"] == country]
                ts = get_time_series(
                    country_data, 
                    country, 
                    selected_product, 
                    selected_flow, 
                    freq='M'
                )
                country_ts_dict[country] = ts
            
            # Plot multiple time series
            fig = plot_multiple_time_series(
                country_ts_dict,
                title=f"Multi-Country Comparison: {selected_product} {selected_flow}",
                xlabel="Date",
                ylabel="Value (KB/D)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate growth rates
            growth_rates = {}
            
            for country, ts in country_ts_dict.items():
                growth_rates[country] = calculate_growth_rate(ts)
            
            # Create growth rate bar chart
            growth_df = pd.DataFrame({
                "Country": list(growth_rates.keys()),
                "CAGR (%)": [growth_rates[country] if growth_rates[country] is not None else 0 for country in growth_rates]
            })
            
            growth_df = growth_df.sort_values("CAGR (%)", ascending=False)
            
            fig = px.bar(
                growth_df,
                x="Country",
                y="CAGR (%)",
                title="Compound Annual Growth Rate",
                color="CAGR (%)",
                color_continuous_scale="RdBu",
                text_auto='.1f'
            )
            
            fig.update_layout(
                xaxis_title="Country",
                yaxis_title="CAGR (%)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Normalized comparison (indexed to 100)
            st.subheader("Normalized Comparison")
            
            # Create normalized time series
            normalized_ts = {}
            
            for country, ts in country_ts_dict.items():
                if len(ts) > 0:
                    base_value = ts.iloc[0]
                    if base_value > 0:
                        normalized_ts[country] = ts / base_value * 100
            
            # Plot normalized time series
            fig = plot_multiple_time_series(
                normalized_ts,
                title="Indexed Comparison (First Period = 100)",
                xlabel="Date",
                ylabel="Index Value"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            st.subheader("Correlation Analysis")
            
            # Create correlation matrix
            corr_df = pd.DataFrame()
            
            for country, ts in country_ts_dict.items():
                corr_df[country] = ts
            
            corr_matrix = corr_df.corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                title="Correlation Between Countries",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Seasonal analysis
            st.subheader("Seasonal Pattern Analysis")
            
            # Choose country for seasonal analysis
            seasonal_country = st.selectbox("Select Country for Seasonal Analysis", multi_countries)
            
            # Calculate monthly averages
            if seasonal_country in country_ts_dict:
                ts = country_ts_dict[seasonal_country]
                
                if len(ts) >= 12:
                    ts_df = pd.DataFrame({"Value": ts})
                    ts_df.index = pd.DatetimeIndex(ts.index)
                    
                    # Get month from index
                    ts_df["Month"] = ts_df.index.month
                    
                    # Calculate monthly averages
                    monthly_avg = ts_df.groupby("Month")["Value"].mean().reset_index()
                    
                    # Map month numbers to names
                    month_names = {
                        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
                    }
                    monthly_avg["Month_Name"] = monthly_avg["Month"].map(month_names)
                    
                    # Create bar chart
                    fig = px.bar(
                        monthly_avg,
                        x="Month_Name",
                        y="Value",
                        title=f"Monthly Averages for {seasonal_country}",
                        color="Value",
                        color_continuous_scale="Viridis",
                        text_auto='.2s'
                    )
                    
                    fig.update_layout(
                        xaxis=dict(
                            categoryorder='array',
                            categoryarray=list(month_names.values())
                        ),
                        xaxis_title="Month",
                        yaxis_title="Average Value (KB/D)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Identify seasonal patterns
                    max_month = monthly_avg.loc[monthly_avg["Value"].idxmax()]
                    min_month = monthly_avg.loc[monthly_avg["Value"].idxmin()]
                    
                    st.markdown(f"""
                    ### Seasonal Insights for {seasonal_country}:
                    
                    - **Peak Month**: {max_month["Month_Name"]} with average value of {max_month["Value"]:,.0f} KB/D
                    - **Lowest Month**: {min_month["Month_Name"]} with average value of {min_month["Value"]:,.0f} KB/D
                    - **Seasonal Amplitude**: {max_month["Value"] - min_month["Value"]:,.0f} KB/D ({(max_month["Value"] - min_month["Value"]) / min_month["Value"] * 100:.1f}% of minimum)
                    """)
                else:
                    st.warning(f"Not enough data for seasonal analysis of {seasonal_country}. Need at least 12 months of data.")
        else:
            # Single country time series analysis
            country_ts = get_time_series(
                filtered_data, 
                selected_country, 
                selected_product, 
                selected_flow, 
                freq='M'
            )
            
            if not country_ts.empty:
                # Plot time series with trend
                st.subheader("Trend Analysis")
                
                # Calculate moving averages
                ts_df = pd.DataFrame({"Value": country_ts})
                ts_df["MA_3"] = ts_df["Value"].rolling(window=3).mean()
                ts_df["MA_6"] = ts_df["Value"].rolling(window=6).mean()
                ts_df["MA_12"] = ts_df["Value"].rolling(window=12).mean()
                
                # Plot with moving averages
                fig = go.Figure()
                
                # Add original time series
                fig.add_trace(
                    go.Scatter(
                        x=ts_df.index,
                        y=ts_df["Value"],
                        mode="lines",
                        name="Original",
                        line=dict(color="blue")
                    )
                )
                
                # Add moving averages
                fig.add_trace(
                    go.Scatter(
                        x=ts_df.index,
                        y=ts_df["MA_3"],
                        mode="lines",
                        name="3-Month MA",
                        line=dict(color="red")
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=ts_df.index,
                        y=ts_df["MA_12"],
                        mode="lines",
                        name="12-Month MA",
                        line=dict(color="green")
                    )
                )
                
                fig.update_layout(
                    title=f"{selected_country} - {selected_product} {selected_flow} with Trend Lines",
                    xaxis_title="Date",
                    yaxis_title="Value (KB/D)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Seasonal analysis
                st.subheader("Seasonal Analysis")
                
                if len(country_ts) >= 12:
                    # Calculate monthly averages
                    ts_df = pd.DataFrame({"Value": country_ts})
                    ts_df.index = pd.DatetimeIndex(country_ts.index)
                    
                    # Get month from index
                    ts_df["Month"] = ts_df.index.month
                    
                    # Calculate monthly averages
                    monthly_avg = ts_df.groupby("Month")["Value"].mean().reset_index()
                    
                    # Map month numbers to names
                    month_names = {
                        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
                    }
                    monthly_avg["Month_Name"] = monthly_avg["Month"].map(month_names)
                    
                    # Create bar chart
                    fig = px.bar(
                        monthly_avg,
                        x="Month_Name",
                        y="Value",
                        title=f"Monthly Averages for {selected_country}",
                        color="Value",
                        color_continuous_scale="Viridis",
                        text_auto='.2s'
                    )
                    
                    fig.update_layout(
                        xaxis=dict(
                            categoryorder='array',
                            categoryarray=list(month_names.values())
                        ),
                        xaxis_title="Month",
                        yaxis_title="Average Value (KB/D)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Year-over-year comparison
                    st.subheader("Year-over-Year Comparison")
                    
                    # Add year column
                    ts_df["Year"] = ts_df.index.year
                    
                    # Create pivot table
                    yoy_pivot = ts_df.pivot_table(
                        values="Value",
                        index="Month",
                        columns="Year",
                        aggfunc="mean"
                    ).reset_index()
                    
                    # Add month names
                    yoy_pivot["Month_Name"] = yoy_pivot["Month"].map(month_names)
                    
                    # Plot YoY comparison
                    fig = go.Figure()
                    
                    for year in sorted(ts_df["Year"].unique()):
                        if year in yoy_pivot.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=yoy_pivot["Month_Name"],
                                    y=yoy_pivot[year],
                                    mode="lines+markers",
                                    name=str(year)
                                )
                            )
                    
                    fig.update_layout(
                        title=f"Year-over-Year Comparison for {selected_country}",
                        xaxis=dict(
                            categoryorder='array',
                            categoryarray=list(month_names.values())
                        ),
                        xaxis_title="Month",
                        yaxis_title="Value (KB/D)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Growth rate calculation
                    st.subheader("Growth Rate Analysis")
                    
                    # Calculate year-over-year growth rates
                    ts_df["YoY_Growth"] = ts_df["Value"].pct_change(periods=12) * 100
                    
                    # Plot growth rates
                    fig = px.line(
                        x=ts_df.index[12:],  # Skip first 12 months (no YoY data)
                        y=ts_df["YoY_Growth"][12:],
                        title=f"Year-over-Year Growth Rate for {selected_country}",
                        labels={"x": "Date", "y": "YoY Growth (%)"}
                    )
                    
                    # Add zero line
                    fig.add_shape(
                        type="line",
                        x0=ts_df.index[12],
                        y0=0,
                        x1=ts_df.index[-1],
                        y1=0,
                        line=dict(color="red", dash="dash")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate average growth rate
                    avg_growth = ts_df["YoY_Growth"][12:].mean()
                    current_growth = ts_df["YoY_Growth"].iloc[-1] if len(ts_df) > 12 else None
                    
                    if current_growth is not None:
                        st.markdown(f"""
                        ### Growth Rate Insights:
                        
                        - **Current Growth Rate**: {current_growth:.1f}% year-over-year
                        - **Average Growth Rate**: {avg_growth:.1f}% over the available period
                        - **Growth Trend**: {"Accelerating" if current_growth > avg_growth else "Decelerating"}
                        """)
                else:
                    st.warning(f"Not enough data for seasonal analysis of {selected_country}. Need at least 12 months of data.")
            else:
                st.warning(f"No data available for {selected_country} - {selected_product} {selected_flow}.")
    
    # Tab 3: Country Comparison
    with tabs[2]:
        st.subheader("Country Comparison")
        
        # Select countries to compare
        if selected_country == "All Countries":
            compare_countries = st.multiselect(
                "Select Countries to Compare",
                countries,
                default=top_countries[:5] if 'top_countries' in locals() else countries[:5]
            )
        else:
            # For single country view, pre-select the current country
            compare_countries = st.multiselect(
                "Select Countries to Compare",
                countries,
                default=[selected_country]
            )
        
        if not compare_countries:
            st.warning("Please select at least one country.")
            return
        
        # Filter data for selected countries
        compare_data = filtered_data[filtered_data["CountryName"].isin(compare_countries)]
        
        # Time aggregation selection
        time_agg = st.radio(
            "Time Aggregation",
            ["Latest Value", "Yearly Average", "Full Time Series"],
            horizontal=True
        )
        
        if time_agg == "Latest Value":
            # Get latest date
            latest_date = compare_data["ReferenceDate"].max()
            
            # Filter for latest date
            latest_compare = compare_data[compare_data["ReferenceDate"] == latest_date]
            
            # Aggregate by country
            country_agg = latest_compare.groupby("CountryName")["ObservedValue"].sum().reset_index()
            country_agg = country_agg.sort_values("ObservedValue", ascending=False)
            
            # Create bar chart
            fig = px.bar(
                country_agg,
                x="CountryName",
                y="ObservedValue",
                title=f"Country Comparison: {selected_product} {selected_flow} ({latest_date.strftime('%b %Y')})",
                color="ObservedValue",
                color_continuous_scale="Viridis",
                text_auto='.2s'
            )
            
            fig.update_layout(
                xaxis_title="Country",
                yaxis_title="Value (KB/D)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.dataframe(
                country_agg.set_index("CountryName").rename(columns={"ObservedValue": "Value (KB/D)"}),
                use_container_width=True
            )
        
        elif time_agg == "Yearly Average":
            # Add year column
            compare_data["Year"] = compare_data["ReferenceDate"].dt.year
            
            # Get unique years
            years = sorted(compare_data["Year"].unique())
            
            # Select year
            selected_year = st.selectbox("Select Year", years, index=len(years)-1 if years else 0)
            
            # Filter for selected year
            year_data = compare_data[compare_data["Year"] == selected_year]
            
            # Aggregate by country
            country_year_agg = year_data.groupby("CountryName")["ObservedValue"].mean().reset_index()
            country_year_agg = country_year_agg.sort_values("ObservedValue", ascending=False)
            
            # Create bar chart
            fig = px.bar(
                country_year_agg,
                x="CountryName",
                y="ObservedValue",
                title=f"Country Comparison: {selected_product} {selected_flow} ({selected_year} Average)",
                color="ObservedValue",
                color_continuous_scale="Viridis",
                text_auto='.2s'
            )
            
            fig.update_layout(
                xaxis_title="Country",
                yaxis_title="Value (KB/D)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.dataframe(
                country_year_agg.set_index("CountryName").rename(columns={"ObservedValue": "Value (KB/D)"}),
                use_container_width=True
            )
            
            # Year-by-year comparison
            st.subheader("Year-by-Year Comparison")
            
            # Create pivot table
            country_year_pivot = compare_data.pivot_table(
                values="ObservedValue",
                index="CountryName",
                columns="Year",
                aggfunc="mean"
            ).reset_index()
            
            # Calculate year-over-year growth
            for i in range(1, len(years)):
                prev_year = years[i-1]
                curr_year = years[i]
                
                if prev_year in country_year_pivot.columns and curr_year in country_year_pivot.columns:
                    growth_col = f"Growth {prev_year}-{curr_year}"
                    country_year_pivot[growth_col] = (
                        (country_year_pivot[curr_year] - country_year_pivot[prev_year]) / 
                        country_year_pivot[prev_year] * 100
                    )
            
            # Format the table
            formatted_pivot = country_year_pivot.copy()
            
            # Format year columns
            for year in years:
                if year in formatted_pivot.columns:
                    formatted_pivot[year] = formatted_pivot[year].map(lambda x: f"{x:,.0f}")
            
            # Format growth columns
            for i in range(1, len(years)):
                prev_year = years[i-1]
                curr_year = years[i]
                growth_col = f"Growth {prev_year}-{curr_year}"
                
                if growth_col in formatted_pivot.columns:
                    formatted_pivot[growth_col] = formatted_pivot[growth_col].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
            
            # Show the table
            st.dataframe(formatted_pivot.set_index("CountryName"), use_container_width=True)
        
        else:  # Full Time Series
            # Create time series for each country
            country_ts_dict = {}
            
            for country in compare_countries:
                country_data = compare_data[compare_data["CountryName"] == country]
                ts = get_time_series(
                    country_data, 
                    country, 
                    selected_product, 
                    selected_flow, 
                    freq='M'
                )
                country_ts_dict[country] = ts
            
            # Plot multiple time series
            fig = plot_multiple_time_series(
                country_ts_dict,
                title=f"Time Series Comparison: {selected_product} {selected_flow}",
                xlabel="Date",
                ylabel="Value (KB/D)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show key statistics
            st.subheader("Key Statistics")
            
            # Calculate statistics for each country
            stats_data = []
            
            for country, ts in country_ts_dict.items():
                if not ts.empty:
                    stats = {
                        "Country": country,
                        "Average": ts.mean(),
                        "Min": ts.min(),
                        "Max": ts.max(),
                        "Current": ts.iloc[-1],
                        "Growth Rate": calculate_growth_rate(ts)
                    }
                    stats_data.append(stats)
            
            # Create DataFrame
            stats_df = pd.DataFrame(stats_data)
            
            # Format the table
            formatted_stats = stats_df.copy()
            formatted_stats["Average"] = formatted_stats["Average"].map(lambda x: f"{x:,.0f}")
            formatted_stats["Min"] = formatted_stats["Min"].map(lambda x: f"{x:,.0f}")
            formatted_stats["Max"] = formatted_stats["Max"].map(lambda x: f"{x:,.0f}")
            formatted_stats["Current"] = formatted_stats["Current"].map(lambda x: f"{x:,.0f}")
            formatted_stats["Growth Rate"] = formatted_stats["Growth Rate"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
            
            # Show the table
            st.dataframe(formatted_stats.set_index("Country"), use_container_width=True)
            
            # Market share analysis
            st.subheader("Market Share Analysis")
            
            # Calculate total for each date
            date_totals = compare_data.groupby("ReferenceDate")["ObservedValue"].sum().reset_index()
            
            # Merge with original data to calculate shares
            share_data = pd.merge(compare_data, date_totals, on="ReferenceDate", suffixes=("", "_total"))
            share_data["Market_Share"] = share_data["ObservedValue"] / share_data["ObservedValue_total"] * 100
            
            # Create time series of market shares
            share_ts_dict = {}
            
            for country in compare_countries:
                country_share = share_data[share_data["CountryName"] == country]
                country_share_ts = get_time_series(
                    country_share, 
                    country, 
                    selected_product, 
                    selected_flow, 
                    freq='M',
                    value_col="Market_Share"
                )
                share_ts_dict[country] = country_share_ts
            
            # Plot market share time series
            fig = plot_multiple_time_series(
                share_ts_dict,
                title="Market Share Evolution",
                xlabel="Date",
                ylabel="Market Share (%)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Flow Analysis
    with tabs[3]:
        st.subheader("Flow Analysis")
        
        # Calculate available flows for the selected product
        available_flows = filtered_data["FlowBreakdown"].unique()
        
        # Get comparable flows
        if selected_country != "All Countries":
            comparable_flows = get_comparable_flows(data, selected_country, selected_product)
        else:
            comparable_flows = get_comparable_flows(data, countries[0], selected_product)
        
        # Flow selection
        selected_flows = st.multiselect(
            "Select Flows to Compare",
            flow_metrics,
            default=[selected_flow] if selected_flow in flow_metrics else []
        )
        
        if not selected_flows:
            st.warning("Please select at least one flow.")
            return
        
        # Country selection for flow analysis
        if selected_country == "All Countries":
            flow_country = st.selectbox("Select Country for Flow Analysis", countries)
        else:
            flow_country = selected_country
        
        # Filter data for selected country and flows
        flow_data = data[
            (data["CountryName"] == flow_country) &
            (data["Product"] == selected_product) &
            (data["FlowBreakdown"].isin(selected_flows)) &
            (data["ReferenceDate"].dt.date >= start_date) &
            (data["ReferenceDate"].dt.date <= end_date)
        ]
        
        # Create time series for each flow
        flow_ts_dict = {}
        
        for flow in selected_flows:
            flow_specific = flow_data[flow_data["FlowBreakdown"] == flow]
            ts = get_time_series(
                flow_specific, 
                flow_country, 
                selected_product, 
                flow, 
                freq='M'
            )
            flow_ts_dict[flow] = ts
        
        # Plot multiple time series
        fig = plot_multiple_time_series(
            flow_ts_dict,
            title=f"{flow_country} - {selected_product} Flow Comparison",
            xlabel="Date",
            ylabel="Value (KB/D)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Flow correlation analysis
        st.subheader("Flow Correlation Analysis")
        
        # Create correlation matrix
        flow_corr_df = pd.DataFrame()
        
        for flow, ts in flow_ts_dict.items():
            if not ts.empty:
                flow_corr_df[flow] = ts
        
        if not flow_corr_df.empty and flow_corr_df.shape[1] > 1:
            corr_matrix = flow_corr_df.corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                title="Correlation Between Flows",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Explain correlations
            st.markdown("""
            ### Correlation Interpretation:
            
            - **+1.0**: Perfect positive correlation (flows move exactly together)
            - **0.0**: No correlation (flows move independently)
            - **-1.0**: Perfect negative correlation (flows move in opposite directions)
            
            Strong positive correlations (> 0.7) indicate flows that tend to increase or decrease together.
            Strong negative correlations (< -0.7) indicate flows that tend to move in opposite directions.
            """)
            
            # Identify strongest correlations
            corr_values = corr_matrix.unstack()
            corr_values = corr_values[corr_values < 1.0]  # Remove self-correlations
            
            top_pos_corr = corr_values.nlargest(3)
            top_neg_corr = corr_values.nsmallest(3)
            
            if not top_pos_corr.empty:
                st.markdown("#### Strongest Positive Correlations:")
                for idx, val in top_pos_corr.items():
                    st.markdown(f"- **{idx[0]}** and **{idx[1]}**: {val:.2f}")
            
            if not top_neg_corr.empty:
                st.markdown("#### Strongest Negative Correlations:")
                for idx, val in top_neg_corr.items():
                    st.markdown(f"- **{idx[0]}** and **{idx[1]}**: {val:.2f}")
        else:
            st.info("Need at least two flows with data for correlation analysis.")
        
        # Flow balance analysis
        if len(selected_flows) >= 2:
            st.subheader("Flow Balance Analysis")
            
            # Select flows for balance calculation
            col1, col2 = st.columns(2)
            
            with col1:
                balance_flow1 = st.selectbox("First Flow", selected_flows, index=0)
            
            with col2:
                balance_flow2 = st.selectbox("Second Flow", 
                                            [f for f in selected_flows if f != balance_flow1], 
                                            index=0)
            
            if balance_flow1 in flow_ts_dict and balance_flow2 in flow_ts_dict:
                # Get time series
                ts1 = flow_ts_dict[balance_flow1]
                ts2 = flow_ts_dict[balance_flow2]
                
                # Align indices
                common_idx = ts1.index.intersection(ts2.index)
                ts1 = ts1.loc[common_idx]
                ts2 = ts2.loc[common_idx]
                
                # Calculate balance
                balance = ts1 - ts2
                
                # Create DataFrame for plotting
                balance_df = pd.DataFrame({
                    "Date": balance.index,
                    balance_flow1: ts1.values,
                    balance_flow2: ts2.values,
                    "Balance": balance.values
                })
                
                # Plot stacked bar chart
                fig = go.Figure()
                
                # Add first flow
                fig.add_trace(
                    go.Bar(
                        x=balance_df["Date"],
                        y=balance_df[balance_flow1],
                        name=balance_flow1
                    )
                )
                
                # Add second flow as negative
                fig.add_trace(
                    go.Bar(
                        x=balance_df["Date"],
                        y=-balance_df[balance_flow2],  # Negative to show as opposite
                        name=balance_flow2
                    )
                )
                
                # Add balance line
                fig.add_trace(
                    go.Scatter(
                        x=balance_df["Date"],
                        y=balance_df["Balance"],
                        mode="lines+markers",
                        name="Balance",
                        line=dict(color="black", width=2)
                    )
                )
                
                fig.update_layout(
                    title=f"Flow Balance: {balance_flow1} vs {balance_flow2}",
                    xaxis_title="Date",
                    yaxis_title="Value (KB/D)",
                    barmode="relative",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate statistics
                avg_balance = balance.mean()
                latest_balance = balance.iloc[-1] if not balance.empty else None
                
                if latest_balance is not None:
                    st.markdown(f"""
                    ### Balance Statistics:
                    
                    - **Current Balance**: {latest_balance:,.0f} KB/D
                    - **Average Balance**: {avg_balance:,.0f} KB/D
                    - **Balance Direction**: {"Surplus" if latest_balance > 0 else "Deficit"} ({balance_flow1} vs {balance_flow2})
                    """)
            else:
                st.warning("Selected flows don't have matching data for balance calculation.")
        
        # Regional flow comparison
        if selected_country == "All Countries":
            st.subheader("Regional Flow Comparison")
            
            # Select flow for regional comparison
            region_flow = st.selectbox("Select Flow for Regional Comparison", flow_metrics, 
                                     index=flow_metrics.index(selected_flow) if selected_flow in flow_metrics else 0)
            
            # Filter data for selected flow
            region_flow_data = filtered_data[filtered_data["FlowBreakdown"] == region_flow]
            
            # Get latest date
            latest_date = region_flow_data["ReferenceDate"].max()
            
            # Filter for latest date
            latest_region_flow = region_flow_data[region_flow_data["ReferenceDate"] == latest_date]
            
            # Create region mapping
            regions = {
                "North America": ["United States", "Canada", "Mexico"],
                "Europe": config.COUNTRY_GROUPS["EUROPE"],
                "Asia Pacific": config.COUNTRY_GROUPS["ASIA_PACIFIC"],
                "Middle East": ["Saudi Arabia", "Iran", "Iraq", "Kuwait", "UAE", "Qatar", "Oman", "Bahrain", "Yemen"],
                "Africa": ["Algeria", "Angola", "Nigeria", "Libya", "Egypt", "South Africa"],
                "Latin America": ["Brazil", "Venezuela", "Colombia", "Argentina", "Ecuador", "Peru", "Chile"]
            }
            
            # Map countries to regions
            country_to_region = {}
            for region, countries_list in regions.items():
                for country in countries_list:
                    country_to_region[country] = region
            
            # Add region column
            latest_region_flow["Region"] = latest_region_flow["CountryName"].map(country_to_region)
            latest_region_flow["Region"].fillna("Other", inplace=True)
            
            # Aggregate by region
            region_agg = latest_region_flow.groupby("Region")["ObservedValue"].sum().reset_index()
            region_agg = region_agg.sort_values("ObservedValue", ascending=False)
            
            # Create bar chart
            fig = px.bar(
                region_agg,
                x="Region",
                y="ObservedValue",
                title=f"Regional Comparison: {selected_product} {region_flow}",
                color="ObservedValue",
                color_continuous_scale="Viridis",
                text_auto='.2s'
            )
            
            fig.update_layout(
                xaxis_title="Region",
                yaxis_title="Value (KB/D)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Data Table
    with tabs[4]:
        st.subheader("Raw Data Explorer")
        
        # Show filter summary
        st.markdown(f"""
        ### Current Filters:
        
        - **Product**: {selected_product}
        - **Flow**: {selected_flow}
        - **Country**: {selected_country}
        - **Date Range**: {start_date} to {end_date}
        """)
        
        # Show data table
        st.dataframe(filtered_data.sort_values(["CountryName", "ReferenceDate"]), use_container_width=True)
        
        # Download option
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{selected_product}_{selected_flow}_{start_date}_to_{end_date}.csv",
            mime="text/csv"
        )

def get_time_series(data, country, product, flow_breakdown, freq='M', value_col="ObservedValue"):
    """
    Helper function to extract time series from data
    
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
    value_col : str
        Column name for values
        
    Returns:
    --------
    time_series : pandas Series
        Time series data with datetime index
    """
    # Filter data
    filtered_data = data.copy()
    
    # Apply filters if not "All"
    if country != "All Countries":
        filtered_data = filtered_data[filtered_data["CountryName"] == country]
    
    if product != "All Products":
        filtered_data = filtered_data[filtered_data["Product"] == product]
    
    if flow_breakdown != "All Flows":
        filtered_data = filtered_data[filtered_data["FlowBreakdown"] == flow_breakdown]
    
    # Group by date
    if freq == 'M':
        filtered_data["Period"] = filtered_data["ReferenceDate"].dt.to_period('M')
    elif freq == 'Q':
        filtered_data["Period"] = filtered_data["ReferenceDate"].dt.to_period('Q')
    elif freq == 'Y':
        filtered_data["Period"] = filtered_data["ReferenceDate"].dt.to_period('Y')
    else:
        filtered_data["Period"] = filtered_data["ReferenceDate"].dt.to_period('M')
    
    # Aggregate
    time_series = filtered_data.groupby("Period")[value_col].mean()
    
    # Convert period index to datetime
    time_series.index = time_series.index.to_timestamp()
    
    # Sort by date
    time_series = time_series.sort_index()
    
    return time_series