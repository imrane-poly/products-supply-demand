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
    Dashboard overview page
    
    Parameters:
    -----------
    data : pandas DataFrame
        Processed data
    """
    st.title("Supply & Demand Dashboard")
    st.write("Get a high-level overview of global oil supply and demand patterns.")
    
    if data.empty:
        st.error("No data available. Please check data loading and filtering.")
        return
    
    # Key filters for dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Product selection
        products = sorted(data["Product"].unique())
        selected_product = st.selectbox(
            "Product", 
            products,
            index=0 if config.DEFAULT_PRODUCT in products else 0
        )
    
    with col2:
        # Country filter with predefined groups
        country_options = ["Global", "Top 10 Producers", "Top 10 Consumers", "OPEC", "Custom Selection"]
        selected_country_option = st.selectbox("Region/Country Group", country_options)
    
    with col3:
        # Date range filter (recent period for dashboard view)
        end_date = data["ReferenceDate"].max().date()
        start_date = end_date - timedelta(days=365)  # Last year by default
        
        period_options = ["Last Year", "Last 2 Years", "Last 5 Years", "All Time"]
        selected_period = st.selectbox("Time Period", period_options)
        
        if selected_period == "Last 2 Years":
            start_date = end_date - timedelta(days=365*2)
        elif selected_period == "Last 5 Years":
            start_date = end_date - timedelta(days=365*5)
        elif selected_period == "All Time":
            start_date = data["ReferenceDate"].min().date()
    
    # Resolve country selection
    if selected_country_option == "Global":
        selected_countries = sorted(data["CountryName"].unique())
    elif selected_country_option == "Top 10 Producers":
        # Get top 10 producers from data
        production_data = data[(data["Product"] == selected_product) & (data["FlowBreakdown"] == "Production")]
        top_producers = production_data.groupby("CountryName")["ObservedValue"].mean().nlargest(10).index.tolist()
        selected_countries = top_producers
    elif selected_country_option == "Top 10 Consumers":
        # Get top 10 consumers from data
        consumption_data = data[(data["Product"] == selected_product) & (data["FlowBreakdown"] == "Consumption")]
        top_consumers = consumption_data.groupby("CountryName")["ObservedValue"].mean().nlargest(10).index.tolist()
        selected_countries = top_consumers
    elif selected_country_option == "OPEC":
        selected_countries = config.COUNTRY_GROUPS["OPEC"]
    else:  # Custom Selection
        all_countries = sorted(data["CountryName"].unique())
        selected_countries = st.multiselect(
            "Select Countries",
            all_countries,
            default=config.DEFAULT_COUNTRIES
        )
    
    # Filter data
    filtered_data = data[
        (data["ReferenceDate"].dt.date >= start_date) &
        (data["ReferenceDate"].dt.date <= end_date) &
        (data["Product"] == selected_product) &
        (data["CountryName"].isin(selected_countries))
    ]
    
    # Create dashboard sections
    st.markdown(f"## {selected_product} Market Overview")
    
    # Create tabs for different views
    tabs = st.tabs([
        "Supply & Demand Balance", 
        "Production Trends", 
        "Consumption Trends", 
        "Import/Export Trends",
        "Market Share"
    ])
    
    # Tab 1: Supply & Demand Balance
    with tabs[0]:
        # Get supply and demand flows
        supply_flows = [flow for flow in data["FlowBreakdown"].unique() if "Production" in flow or "Import" in flow]
        demand_flows = [flow for flow in data["FlowBreakdown"].unique() if "Consumption" in flow or "Export" in flow]
        
        # Filter for supply and demand data
        supply_data = filtered_data[filtered_data["FlowBreakdown"].isin(supply_flows)]
        demand_data = filtered_data[filtered_data["FlowBreakdown"].isin(demand_flows)]
        
        # Aggregate by date
        supply_by_date = supply_data.groupby("ReferenceDate")["ObservedValue"].sum().reset_index()
        demand_by_date = demand_data.groupby("ReferenceDate")["ObservedValue"].sum().reset_index()
        
        # Create data for balance chart
        balance_data = pd.merge(
            supply_by_date, 
            demand_by_date, 
            on="ReferenceDate", 
            suffixes=("_supply", "_demand")
        )
        
        # Calculate balance
        balance_data["Balance"] = balance_data["ObservedValue_supply"] - balance_data["ObservedValue_demand"]
        
        # Create figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add supply
        fig.add_trace(
            go.Bar(
                x=balance_data["ReferenceDate"],
                y=balance_data["ObservedValue_supply"],
                name="Supply",
                marker_color="green",
                opacity=0.7
            ),
            secondary_y=False
        )
        
        # Add demand
        fig.add_trace(
            go.Bar(
                x=balance_data["ReferenceDate"],
                y=balance_data["ObservedValue_demand"],
                name="Demand",
                marker_color="red",
                opacity=0.7
            ),
            secondary_y=False
        )
        
        # Add balance line
        fig.add_trace(
            go.Scatter(
                x=balance_data["ReferenceDate"],
                y=balance_data["Balance"],
                name="Balance",
                line=dict(color="blue", width=2)
            ),
            secondary_y=True
        )
        
        # Add zero line for balance
        fig.add_shape(
            type="line",
            x0=balance_data["ReferenceDate"].min(),
            y0=0,
            x1=balance_data["ReferenceDate"].max(),
            y1=0,
            line=dict(color="gray", dash="dash"),
            yref="y2"
        )
        
        # Update layout
        fig.update_layout(
            title=f"{selected_product} Supply & Demand Balance",
            barmode="group",
            height=500
        )
        
        fig.update_yaxes(title_text="Supply/Demand (KB/D)", secondary_y=False)
        fig.update_yaxes(title_text="Balance (KB/D)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics
        if not balance_data.empty:
            # Get latest data
            latest_balance = balance_data.iloc[-1]
            avg_balance = balance_data["Balance"].mean()
            
            # Create metrics
            metrics = {
                "Latest Supply": f"{latest_balance['ObservedValue_supply']:,.0f} KB/D",
                "Latest Demand": f"{latest_balance['ObservedValue_demand']:,.0f} KB/D",
                "Latest Balance": f"{latest_balance['Balance']:,.0f} KB/D",
                "Avg Balance": f"{avg_balance:,.0f} KB/D"
            }
            
            display_metric_row(metrics)
            
            # Balance insights
            balance_status = "surplus" if latest_balance["Balance"] > 0 else "deficit"
            balance_trend = "increasing" if balance_data["Balance"].iloc[-3:].mean() > balance_data["Balance"].iloc[-6:-3].mean() else "decreasing"
            
            st.markdown(f"""
            ### Market Balance Insights
            
            The market is currently in **{balance_status}** with a {balance_trend} trend. 
            This suggests {"upward pressure on inventories and potentially downward pressure on prices" if balance_status == "surplus" else "inventory draws and potentially upward pressure on prices"}.
            """)
        
        # Regional breakdown
        st.subheader("Regional Supply & Demand")
        
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
        
        # Add region column
        supply_data_with_region = supply_data.copy()
        demand_data_with_region = demand_data.copy()
        
        supply_data_with_region["Region"] = supply_data_with_region["CountryName"].map(country_to_region)
        demand_data_with_region["Region"] = demand_data_with_region["CountryName"].map(country_to_region)
        
        # Fill NA regions
        supply_data_with_region["Region"].fillna("Other", inplace=True)
        demand_data_with_region["Region"].fillna("Other", inplace=True)
        
        # Get latest date
        latest_date = filtered_data["ReferenceDate"].max()
        
        # Filter for latest date
        latest_supply = supply_data_with_region[supply_data_with_region["ReferenceDate"] == latest_date]
        latest_demand = demand_data_with_region[demand_data_with_region["ReferenceDate"] == latest_date]
        
        # Aggregate by region
        region_supply = latest_supply.groupby("Region")["ObservedValue"].sum().reset_index()
        region_demand = latest_demand.groupby("Region")["ObservedValue"].sum().reset_index()
        
        # Create region balance data
        region_balance = pd.merge(
            region_supply,
            region_demand,
            on="Region",
            suffixes=("_supply", "_demand")
        )
        
        # Calculate balance
        region_balance["Balance"] = region_balance["ObservedValue_supply"] - region_balance["ObservedValue_demand"]
        
        # Sort by absolute balance
        region_balance = region_balance.sort_values("Balance", key=abs, ascending=False)
        
        # Create stacked bar chart
        fig = go.Figure()
        
        # Add supply bars
        fig.add_trace(
            go.Bar(
                x=region_balance["Region"],
                y=region_balance["ObservedValue_supply"],
                name="Supply",
                marker_color="green",
                opacity=0.7
            )
        )
        
        # Add demand bars
        fig.add_trace(
            go.Bar(
                x=region_balance["Region"],
                y=-region_balance["ObservedValue_demand"],  # Negative to show as opposite
                name="Demand",
                marker_color="red",
                opacity=0.7
            )
        )
        
        # Add balance points
        fig.add_trace(
            go.Scatter(
                x=region_balance["Region"],
                y=region_balance["Balance"],
                mode="markers",
                name="Balance",
                marker=dict(
                    color="blue",
                    size=10,
                    line=dict(width=2, color="DarkSlateGrey")
                )
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Regional Supply & Demand Balance",
            barmode="relative",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Regional insights
        surplus_regions = region_balance[region_balance["Balance"] > 0]["Region"].tolist()
        deficit_regions = region_balance[region_balance["Balance"] < 0]["Region"].tolist()
        
        if surplus_regions and deficit_regions:
            st.markdown(f"""
            ### Regional Flow Insights
            
            **Net Exporting Regions**: {", ".join(surplus_regions[:3])}
            
            **Net Importing Regions**: {", ".join(deficit_regions[:3])}
            
            This suggests potential for trade flows from {surplus_regions[0]} to {deficit_regions[0]}.
            """)
    
    # Tab 2: Production Trends
    with tabs[1]:
        # Filter production data
        production_data = filtered_data[filtered_data["FlowBreakdown"] == "Production"]
        
        if production_data.empty:
            st.info(f"No production data available for {selected_product}.")
        else:
            # Top producers
            st.subheader("Top Producers")
            
            # Get latest date
            latest_date = production_data["ReferenceDate"].max()
            
            # Filter for latest date
            latest_production = production_data[production_data["ReferenceDate"] == latest_date]
            
            # Aggregate by country
            country_production = latest_production.groupby("CountryName")["ObservedValue"].sum().reset_index()
            
            # Sort by production
            country_production = country_production.sort_values("ObservedValue", ascending=False)
            
            # Take top 10
            top_producers = country_production.head(10)
            
            # Create bar chart
            fig = px.bar(
                top_producers,
                x="CountryName",
                y="ObservedValue",
                title=f"Top 10 {selected_product} Producers",
                color="ObservedValue",
                color_continuous_scale="Viridis",
                text_auto='.2s'
            )
            
            fig.update_layout(
                xaxis_title="Country",
                yaxis_title="Production (KB/D)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Production growth rates
            st.subheader("Production Growth Rates")
            
            # Get time series for each country
            country_ts_dict = {}
            
            for country in top_producers["CountryName"].tolist():
                country_data = production_data[production_data["CountryName"] == country]
                ts = get_time_series(
                    country_data, 
                    country, 
                    selected_product, 
                    "Production", 
                    freq='M'
                )
                country_ts_dict[country] = ts
            
            # Calculate growth rates
            growth_rates = {}
            
            for country, ts in country_ts_dict.items():
                # Check if we have enough data for YoY growth
                if len(ts) >= 13:
                    # Calculate YoY growth
                    latest = ts.iloc[-1]
                    year_ago = ts.iloc[-13]
                    
                    if year_ago > 0:
                        growth = (latest - year_ago) / year_ago * 100
                        growth_rates[country] = growth
            
            # Create growth rate chart
            if growth_rates:
                growth_df = pd.DataFrame({
                    "Country": list(growth_rates.keys()),
                    "YoY Growth (%)": list(growth_rates.values())
                })
                
                growth_df = growth_df.sort_values("YoY Growth (%)", ascending=False)
                
                fig = px.bar(
                    growth_df,
                    x="Country",
                    y="YoY Growth (%)",
                    title="Year-over-Year Production Growth",
                    color="YoY Growth (%)",
                    color_continuous_scale="RdBu",
                    text_auto='.1f'
                )
                
                # Add zero line
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    y0=0,
                    x1=len(growth_df) - 0.5,
                    y1=0,
                    line=dict(color="gray", dash="dash")
                )
                
                fig.update_layout(
                    xaxis_title="Country",
                    yaxis_title="YoY Growth (%)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Growth insights
                fastest_growing = growth_df.iloc[0]["Country"] if not growth_df.empty else None
                fastest_declining = growth_df.iloc[-1]["Country"] if not growth_df.empty and growth_df.iloc[-1]["YoY Growth (%)"] < 0 else None
                
                if fastest_growing and fastest_declining:
                    st.markdown(f"""
                    ### Production Growth Insights
                    
                    **Fastest Growing Producer**: {fastest_growing} (+{growth_df.iloc[0]['YoY Growth (%)']:.1f}%)
                    
                    **Fastest Declining Producer**: {fastest_declining} ({growth_df.iloc[-1]['YoY Growth (%)']:.1f}%)
                    
                    This suggests a potential shift in production capacity and market influence.
                    """)
            
            # Production time series
            st.subheader("Production Time Series")
            
            # Top 5 producers for time series
            top_5_producers = top_producers.head(5)["CountryName"].tolist()
            
            # Create time series dict for top 5
            top_5_ts = {}
            
            for country in top_5_producers:
                if country in country_ts_dict:
                    top_5_ts[country] = country_ts_dict[country]
            
            # Plot time series
            fig = plot_multiple_time_series(
                top_5_ts,
                title=f"Top 5 {selected_product} Producers",
                xlabel="Date",
                ylabel="Production (KB/D)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Total production trend
            st.subheader("Total Production Trend")
            
            # Aggregate production by date
            total_production = production_data.groupby("ReferenceDate")["ObservedValue"].sum().reset_index()
            
            # Create total production series
            total_production_ts = pd.Series(
                total_production["ObservedValue"].values,
                index=total_production["ReferenceDate"]
            )
            
            # Plot time series
            fig = plot_time_series(
                total_production_ts,
                title=f"Total {selected_product} Production",
                xlabel="Date",
                ylabel="Production (KB/D)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Production trend insights
            if len(total_production_ts) >= 6:
                recent_avg = total_production_ts.iloc[-3:].mean()
                previous_avg = total_production_ts.iloc[-6:-3].mean()
                
                trend_pct = (recent_avg - previous_avg) / previous_avg * 100
                trend_direction = "increasing" if trend_pct > 0 else "decreasing"
                
                st.markdown(f"""
                ### Production Trend Insights
                
                Total production is **{trend_direction}** by {abs(trend_pct):.1f}% when comparing the most recent 3 months to the previous 3 months.
                
                This suggests {"potential oversupply if demand remains constant" if trend_direction == "increasing" else "potential tightening of supply if demand remains constant"}.
                """)
    
    # Tab 3: Consumption Trends
    with tabs[2]:
        # Filter consumption data
        consumption_data = filtered_data[filtered_data["FlowBreakdown"] == "Consumption"]
        
        if consumption_data.empty:
            st.info(f"No consumption data available for {selected_product}.")
        else:
            # Top consumers
            st.subheader("Top Consumers")
            
            # Get latest date
            latest_date = consumption_data["ReferenceDate"].max()
            
            # Filter for latest date
            latest_consumption = consumption_data[consumption_data["ReferenceDate"] == latest_date]
            
            # Aggregate by country
            country_consumption = latest_consumption.groupby("CountryName")["ObservedValue"].sum().reset_index()
            
            # Sort by consumption
            country_consumption = country_consumption.sort_values("ObservedValue", ascending=False)
            
            # Take top 10
            top_consumers = country_consumption.head(10)
            
            # Create bar chart
            fig = px.bar(
                top_consumers,
                x="CountryName",
                y="ObservedValue",
                title=f"Top 10 {selected_product} Consumers",
                color="ObservedValue",
                color_continuous_scale="Viridis",
                text_auto='.2s'
            )
            
            fig.update_layout(
                xaxis_title="Country",
                yaxis_title="Consumption (KB/D)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Consumption growth rates
            st.subheader("Consumption Growth Rates")
            
            # Get time series for each country
            country_ts_dict = {}
            
            for country in top_consumers["CountryName"].tolist():
                country_data = consumption_data[consumption_data["CountryName"] == country]
                ts = get_time_series(
                    country_data, 
                    country, 
                    selected_product, 
                    "Consumption", 
                    freq='M'
                )
                country_ts_dict[country] = ts
            
            # Calculate growth rates
            growth_rates = {}
            
            for country, ts in country_ts_dict.items():
                # Check if we have enough data for YoY growth
                if len(ts) >= 13:
                    # Calculate YoY growth
                    latest = ts.iloc[-1]
                    year_ago = ts.iloc[-13]
                    
                    if year_ago > 0:
                        growth = (latest - year_ago) / year_ago * 100
                        growth_rates[country] = growth
            
            # Create growth rate chart
            if growth_rates:
                growth_df = pd.DataFrame({
                    "Country": list(growth_rates.keys()),
                    "YoY Growth (%)": list(growth_rates.values())
                })
                
                growth_df = growth_df.sort_values("YoY Growth (%)", ascending=False)
                
                fig = px.bar(
                    growth_df,
                    x="Country",
                    y="YoY Growth (%)",
                    title="Year-over-Year Consumption Growth",
                    color="YoY Growth (%)",
                    color_continuous_scale="RdBu",
                    text_auto='.1f'
                )
                
                # Add zero line
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    y0=0,
                    x1=len(growth_df) - 0.5,
                    y1=0,
                    line=dict(color="gray", dash="dash")
                )
                
                fig.update_layout(
                    xaxis_title="Country",
                    yaxis_title="YoY Growth (%)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Growth insights
                fastest_growing = growth_df.iloc[0]["Country"] if not growth_df.empty else None
                fastest_declining = growth_df.iloc[-1]["Country"] if not growth_df.empty and growth_df.iloc[-1]["YoY Growth (%)"] < 0 else None
                
                if fastest_growing and fastest_declining:
                    st.markdown(f"""
                    ### Consumption Growth Insights
                    
                    **Fastest Growing Consumer**: {fastest_growing} (+{growth_df.iloc[0]['YoY Growth (%)']:.1f}%)
                    
                    **Fastest Declining Consumer**: {fastest_declining} ({growth_df.iloc[-1]['YoY Growth (%)']:.1f}%)
                    
                    This suggests shifts in economic activity and energy demand patterns.
                    """)
            
            # Consumption time series
            st.subheader("Consumption Time Series")
            
            # Top 5 consumers for time series
            top_5_consumers = top_consumers.head(5)["CountryName"].tolist()
            
            # Create time series dict for top 5
            top_5_ts = {}
            
            for country in top_5_consumers:
                if country in country_ts_dict:
                    top_5_ts[country] = country_ts_dict[country]
            
            # Plot time series
            fig = plot_multiple_time_series(
                top_5_ts,
                title=f"Top 5 {selected_product} Consumers",
                xlabel="Date",
                ylabel="Consumption (KB/D)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Total consumption trend
            st.subheader("Total Consumption Trend")
            
            # Aggregate consumption by date
            total_consumption = consumption_data.groupby("ReferenceDate")["ObservedValue"].sum().reset_index()
            
            # Create total consumption series
            total_consumption_ts = pd.Series(
                total_consumption["ObservedValue"].values,
                index=total_consumption["ReferenceDate"]
            )
            
            # Plot time series
            fig = plot_time_series(
                total_consumption_ts,
                title=f"Total {selected_product} Consumption",
                xlabel="Date",
                ylabel="Consumption (KB/D)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Consumption trend insights
            if len(total_consumption_ts) >= 6:
                recent_avg = total_consumption_ts.iloc[-3:].mean()
                previous_avg = total_consumption_ts.iloc[-6:-3].mean()
                
                trend_pct = (recent_avg - previous_avg) / previous_avg * 100
                trend_direction = "increasing" if trend_pct > 0 else "decreasing"
                
                st.markdown(f"""
                ### Consumption Trend Insights
                
                Total consumption is **{trend_direction}** by {abs(trend_pct):.1f}% when comparing the most recent 3 months to the previous 3 months.
                
                This suggests {"potential tightening of supply if production remains constant" if trend_direction == "increasing" else "potential oversupply if production remains constant"}.
                """)
            
            # Seasonal pattern analysis
            st.subheader("Seasonal Consumption Patterns")
            
            if len(total_consumption_ts) >= 24:
                # Create DataFrame
                consumption_df = pd.DataFrame({"Consumption": total_consumption_ts})
                
                # Extract month
                consumption_df["Month"] = consumption_df.index.month
                
                # Calculate monthly averages
                monthly_avg = consumption_df.groupby("Month")["Consumption"].mean().reset_index()
                
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
                    y="Consumption",
                    title=f"Monthly Consumption Pattern for {selected_product}",
                    color="Consumption",
                    color_continuous_scale="Viridis",
                    text_auto='.2s'
                )
                
                fig.update_layout(
                    xaxis=dict(
                        categoryorder='array',
                        categoryarray=list(month_names.values())
                    ),
                    xaxis_title="Month",
                    yaxis_title="Average Consumption (KB/D)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Seasonal insights
                max_month = monthly_avg.loc[monthly_avg["Consumption"].idxmax()]
                min_month = monthly_avg.loc[monthly_avg["Consumption"].idxmin()]
                
                st.markdown(f"""
                ### Seasonal Consumption Insights
                
                **Peak Consumption Month**: {max_month["Month_Name"]} with average of {max_month["Consumption"]:,.0f} KB/D
                
                **Lowest Consumption Month**: {min_month["Month_Name"]} with average of {min_month["Consumption"]:,.0f} KB/D
                
                **Seasonal Amplitude**: {max_month["Consumption"] - min_month["Consumption"]:,.0f} KB/D ({(max_month["Consumption"] - min_month["Consumption"]) / min_month["Consumption"] * 100:.1f}% of minimum)
                
                This seasonal pattern should be considered when forecasting future demand.
                """)
    
    # Tab 4: Import/Export Trends
    with tabs[3]:
        # Filter import and export data
        import_data = filtered_data[filtered_data["FlowBreakdown"] == "Imports"]
        export_data = filtered_data[filtered_data["FlowBreakdown"] == "Exports"]
        
        if import_data.empty and export_data.empty:
            st.info(f"No import/export data available for {selected_product}.")
        else:
            # Net importers and exporters
            st.subheader("Net Importers and Exporters")
            
            # Get latest date
            latest_date = filtered_data["ReferenceDate"].max()
            
            # Latest import and export data
            latest_imports = import_data[import_data["ReferenceDate"] == latest_date]
            latest_exports = export_data[export_data["ReferenceDate"] == latest_date]
            
            # Aggregate by country
            country_imports = latest_imports.groupby("CountryName")["ObservedValue"].sum().reset_index()
            country_exports = latest_exports.groupby("CountryName")["ObservedValue"].sum().reset_index()
            
            # Merge import and export data
            trade_balance = pd.merge(
                country_imports,
                country_exports,
                on="CountryName",
                how="outer",
                suffixes=("_import", "_export")
            )
            
            # Fill NAs with zeros
            trade_balance.fillna(0, inplace=True)
            
            # Calculate net trade
            trade_balance["Net_Trade"] = trade_balance["ObservedValue_export"] - trade_balance["ObservedValue_import"]
            
            # Sort by net trade
            trade_balance = trade_balance.sort_values("Net_Trade", ascending=False)
            
            # Split into net exporters and importers
            net_exporters = trade_balance[trade_balance["Net_Trade"] > 0].head(10)
            net_importers = trade_balance[trade_balance["Net_Trade"] < 0].sort_values("Net_Trade").head(10)
            
            # Create stacked bar charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Net exporters chart
                fig = go.Figure()
                
                # Add export bars
                fig.add_trace(
                    go.Bar(
                        x=net_exporters["CountryName"],
                        y=net_exporters["ObservedValue_export"],
                        name="Exports",
                        marker_color="green"
                    )
                )
                
                # Add import bars
                fig.add_trace(
                    go.Bar(
                        x=net_exporters["CountryName"],
                        y=-net_exporters["ObservedValue_import"],
                        name="Imports",
                        marker_color="red"
                    )
                )
                
                # Add net trade line
                fig.add_trace(
                    go.Scatter(
                        x=net_exporters["CountryName"],
                        y=net_exporters["Net_Trade"],
                        mode="markers",
                        name="Net Trade",
                        marker=dict(color="blue", size=10)
                    )
                )
                
                fig.update_layout(
                    title="Top Net Exporters",
                    xaxis_title="Country",
                    yaxis_title="KB/D",
                    barmode="relative",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Net importers chart
                fig = go.Figure()
                
                # Add export bars
                fig.add_trace(
                    go.Bar(
                        x=net_importers["CountryName"],
                        y=net_importers["ObservedValue_export"],
                        name="Exports",
                        marker_color="green"
                    )
                )
                
                # Add import bars
                fig.add_trace(
                    go.Bar(
                        x=net_importers["CountryName"],
                        y=-net_importers["ObservedValue_import"],
                        name="Imports",
                        marker_color="red"
                    )
                )
                
                # Add net trade line
                fig.add_trace(
                    go.Scatter(
                        x=net_importers["CountryName"],
                        y=net_importers["Net_Trade"],
                        mode="markers",
                        name="Net Trade",
                        marker=dict(color="blue", size=10)
                    )
                )
                
                fig.update_layout(
                    title="Top Net Importers",
                    xaxis_title="Country",
                    yaxis_title="KB/D",
                    barmode="relative",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Global trade trends
            st.subheader("Global Trade Trends")
            
            # Aggregate imports and exports by date
            imports_by_date = import_data.groupby("ReferenceDate")["ObservedValue"].sum().reset_index()
            exports_by_date = export_data.groupby("ReferenceDate")["ObservedValue"].sum().reset_index()
            
            # Merge
            global_trade = pd.merge(
                imports_by_date,
                exports_by_date,
                on="ReferenceDate",
                how="outer",
                suffixes=("_import", "_export")
            )
            
            # Fill NAs with zeros
            global_trade.fillna(0, inplace=True)
            
            # Sort by date
            global_trade = global_trade.sort_values("ReferenceDate")
            
            # Create line chart
            fig = go.Figure()
            
            # Add imports
            fig.add_trace(
                go.Scatter(
                    x=global_trade["ReferenceDate"],
                    y=global_trade["ObservedValue_import"],
                    mode="lines",
                    name="Global Imports",
                    line=dict(color="red")
                )
            )
            
            # Add exports
            fig.add_trace(
                go.Scatter(
                    x=global_trade["ReferenceDate"],
                    y=global_trade["ObservedValue_export"],
                    mode="lines",
                    name="Global Exports",
                    line=dict(color="green")
                )
            )
            
            fig.update_layout(
                title=f"Global {selected_product} Trade",
                xaxis_title="Date",
                yaxis_title="Volume (KB/D)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade flow matrix
            st.subheader("Trade Flow Matrix")
            
            # Create a dummy trade flow matrix for demonstration
            # In a real app, this would use actual origin-destination data
            
            # Select top trading countries
            top_trading = pd.concat([
                net_exporters["CountryName"].head(5),
                net_importers["CountryName"].head(5)
            ]).unique()
            
            # Create dummy matrix
            flow_matrix = pd.DataFrame(0, index=top_trading, columns=top_trading)
            
            # Fill with random data for exporters to importers
            for exporter in net_exporters["CountryName"].head(5):
                for importer in net_importers["CountryName"].head(5):
                    if exporter in flow_matrix.index and importer in flow_matrix.columns:
                        # Proportional to export and import volumes
                        exp_volume = net_exporters[net_exporters["CountryName"] == exporter]["ObservedValue_export"].iloc[0]
                        imp_volume = net_importers[net_importers["CountryName"] == importer]["ObservedValue_import"].iloc[0]
                        
                        # Simplified flow estimate
                        flow = min(exp_volume, imp_volume) * np.random.uniform(0.1, 0.3)
                        flow_matrix.loc[exporter, importer] = flow
            
            # Create heatmap
            fig = px.imshow(
                flow_matrix,
                labels=dict(x="Importer", y="Exporter", color="Flow (KB/D)"),
                x=flow_matrix.columns,
                y=flow_matrix.index,
                title="Estimated Trade Flows",
                color_continuous_scale="Viridis"
            )
            
            fig.update_layout(height=600)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade insights
            st.markdown("""
            ### Trade Flow Insights
            
            The matrix above shows estimated trade flows between major exporters and importers.
            Darker colors indicate higher volume flows.
            
            This visualization helps identify key trading relationships and potential market disruptions.
            
            *Note: This is a simplified representation. Actual trade flows would require detailed
            origin-destination data.*
            """)
    
    # Tab 5: Market Share
    with tabs[4]:
        st.subheader("Market Share Analysis")
        
        # Production market share
        production_data = filtered_data[filtered_data["FlowBreakdown"] == "Production"]
        consumption_data = filtered_data[filtered_data["FlowBreakdown"] == "Consumption"]
        
        if not production_data.empty:
            # Get latest date
            latest_date = production_data["ReferenceDate"].max()
            
            # Filter for latest date
            latest_production = production_data[production_data["ReferenceDate"] == latest_date]
            
            # Calculate total
            total_production = latest_production["ObservedValue"].sum()
            
            # Calculate shares
            production_shares = latest_production.copy()
            production_shares["Market_Share"] = production_shares["ObservedValue"] / total_production * 100
            
            # Sort by share
            production_shares = production_shares.sort_values("Market_Share", ascending=False)
            
            # Top 10 for pie chart
            top_10_production = production_shares.head(10)
            
            # Group rest as "Others"
            if len(production_shares) > 10:
                others_share = production_shares.iloc[10:]["Market_Share"].sum()
                others_row = pd.DataFrame({
                    "CountryName": ["Others"],
                    "ObservedValue": [production_shares.iloc[10:]["ObservedValue"].sum()],
                    "Market_Share": [others_share]
                })
                
                top_10_production = pd.concat([top_10_production[["CountryName", "ObservedValue", "Market_Share"]], others_row])
            
            # Create pie chart
            fig = px.pie(
                top_10_production,
                values="Market_Share",
                names="CountryName",
                title=f"{selected_product} Production Market Share",
                hover_data=["ObservedValue"],
                labels={"ObservedValue": "Production (KB/D)"}
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Market concentration metrics
            st.subheader("Market Concentration")
            
            # Calculate Herfindahl-Hirschman Index (HHI)
            hhi = (production_shares["Market_Share"] ** 2).sum() / 100  # Divide by 100 to scale percentages
            
            # Calculate CR4 (4-firm concentration ratio)
            cr4 = production_shares.head(4)["Market_Share"].sum()
            
            # Interpret concentration
            if hhi > 25:
                concentration = "highly concentrated"
            elif hhi > 15:
                concentration = "moderately concentrated"
            else:
                concentration = "competitive"
            
            # Create metrics
            metrics = {
                "HHI": f"{hhi:.1f}",
                "CR4": f"{cr4:.1f}%",
                "Top Producer": production_shares.iloc[0]["CountryName"],
                "Top Producer Share": f"{production_shares.iloc[0]['Market_Share']:.1f}%"
            }
            
            display_metric_row(metrics)
            
            st.markdown(f"""
            ### Market Concentration Insights
            
            The {selected_product} production market is **{concentration}** with an HHI of {hhi:.1f}.
            
            The top 4 producers control {cr4:.1f}% of global production, with {production_shares.iloc[0]['CountryName']}
            being the largest producer at {production_shares.iloc[0]['Market_Share']:.1f}% market share.
            
            This indicates {"potential for market power and price influence" if concentration != "competitive" else "a diverse supply base with limited individual market power"}.
            """)
        
        # Market share evolution
        st.subheader("Market Share Evolution")
        
        if not production_data.empty:
            # Select top countries for visualization
            top_countries = production_shares.head(5)["CountryName"].tolist()
            
            # Filter for these countries
            top_countries_data = production_data[production_data["CountryName"].isin(top_countries)]
            
            # Calculate total by date
            total_by_date = production_data.groupby("ReferenceDate")["ObservedValue"].sum().reset_index()
            
            # Merge to calculate shares
            shares_data = pd.merge(
                top_countries_data,
                total_by_date,
                on="ReferenceDate",
                suffixes=("", "_total")
            )
            
            # Calculate shares
            shares_data["Market_Share"] = shares_data["ObservedValue"] / shares_data["ObservedValue_total"] * 100
            
            # Create share evolution chart
            fig = px.line(
                shares_data,
                x="ReferenceDate",
                y="Market_Share",
                color="CountryName",
                title=f"Evolution of {selected_product} Production Market Share",
                labels={"Market_Share": "Market Share (%)", "ReferenceDate": "Date"}
            )
            
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate share changes
            if len(shares_data["ReferenceDate"].unique()) >= 2:
                # Get earliest and latest dates
                earliest_date = shares_data["ReferenceDate"].min()
                latest_date = shares_data["ReferenceDate"].max()
                
                # Filter for these dates
                earliest_shares = shares_data[shares_data["ReferenceDate"] == earliest_date]
                latest_shares = shares_data[shares_data["ReferenceDate"] == latest_date]
                
                # Merge to calculate changes
                share_changes = pd.merge(
                    earliest_shares[["CountryName", "Market_Share"]],
                    latest_shares[["CountryName", "Market_Share"]],
                    on="CountryName",
                    suffixes=("_start", "_end")
                )
                
                # Calculate change
                share_changes["Change"] = share_changes["Market_Share_end"] - share_changes["Market_Share_start"]
                
                # Sort by change
                share_changes = share_changes.sort_values("Change", ascending=False)
                
                # Create bar chart
                fig = px.bar(
                    share_changes,
                    x="CountryName",
                    y="Change",
                    title=f"Market Share Change ({earliest_date.strftime('%b %Y')} to {latest_date.strftime('%b %Y')})",
                    color="Change",
                    color_continuous_scale="RdBu",
                    text_auto='.1f'
                )
                
                # Add zero line
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    y0=0,
                    x1=len(share_changes) - 0.5,
                    y1=0,
                    line=dict(color="gray", dash="dash")
                )
                
                fig.update_layout(
                    xaxis_title="Country",
                    yaxis_title="Market Share Change (percentage points)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Share change insights
                biggest_gainer = share_changes.iloc[0]["CountryName"] if not share_changes.empty and share_changes.iloc[0]["Change"] > 0 else None
                biggest_loser = share_changes.iloc[-1]["CountryName"] if not share_changes.empty and share_changes.iloc[-1]["Change"] < 0 else None
                
                if biggest_gainer and biggest_loser:
                    st.markdown(f"""
                    ### Market Share Shift Insights
                    
                    **Biggest Market Share Gainer**: {biggest_gainer} (+{share_changes.iloc[0]['Change']:.1f} percentage points)
                    
                    **Biggest Market Share Loser**: {biggest_loser} ({share_changes.iloc[-1]['Change']:.1f} percentage points)
                    
                    This indicates a shift in market influence from {biggest_loser} to {biggest_gainer}.
                    """)
        
        # Consumption market share
        if not consumption_data.empty:
            st.subheader("Consumption Market Share")
            
            # Get latest date
            latest_date = consumption_data["ReferenceDate"].max()
            
            # Filter for latest date
            latest_consumption = consumption_data[consumption_data["ReferenceDate"] == latest_date]
            
            # Calculate total
            total_consumption = latest_consumption["ObservedValue"].sum()
            
            # Calculate shares
            consumption_shares = latest_consumption.copy()
            consumption_shares["Market_Share"] = consumption_shares["ObservedValue"] / total_consumption * 100
            
            # Sort by share
            consumption_shares = consumption_shares.sort_values("Market_Share", ascending=False)
            
            # Top 10 for pie chart
            top_10_consumption = consumption_shares.head(10)
            
            # Group rest as "Others"
            if len(consumption_shares) > 10:
                others_share = consumption_shares.iloc[10:]["Market_Share"].sum()
                others_row = pd.DataFrame({
                    "CountryName": ["Others"],
                    "ObservedValue": [consumption_shares.iloc[10:]["ObservedValue"].sum()],
                    "Market_Share": [others_share]
                })
                
                top_10_consumption = pd.concat([top_10_consumption[["CountryName", "ObservedValue", "Market_Share"]], others_row])
            
            # Create pie chart
            fig = px.pie(
                top_10_consumption,
                values="Market_Share",
                names="CountryName",
                title=f"{selected_product} Consumption Market Share",
                hover_data=["ObservedValue"],
                labels={"ObservedValue": "Consumption (KB/D)"}
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Per capita consumption
            st.subheader("Per Capita Consumption")
            
            # Create dummy population data
            # In a real app, you would use actual population data
            populations = {
                "United States": 331002651,
                "China": 1411778724,
                "India": 1380004385,
                "Japan": 126476461,
                "Germany": 83783942,
                "Russia": 145934462,
                "South Korea": 51269185,
                "Brazil": 212559417,
                "Canada": 37742154,
                "France": 65273511,
                "United Kingdom": 67886011,
                "Italy": 60461826,
                "Mexico": 128932753,
                "Indonesia": 273523615,
                "Saudi Arabia": 34813871
            }
            
            # Add population data
            per_capita_data = consumption_shares.copy()
            per_capita_data["Population"] = per_capita_data["CountryName"].map(populations)
            
            # Calculate per capita (convert KB/D to barrels per year per person)
            per_capita_data["Per_Capita"] = per_capita_data.apply(
                lambda row: row["ObservedValue"] * 1000 * 365 / row["Population"] if row["Population"] > 0 else 0, 
                axis=1
            )
            
            # Filter rows with population data
            per_capita_data = per_capita_data.dropna(subset=["Population"])
            
            # Sort by per capita
            per_capita_data = per_capita_data.sort_values("Per_Capita", ascending=False)
            
            # Create bar chart
            fig = px.bar(
                per_capita_data.head(15),
                x="CountryName",
                y="Per_Capita",
                title=f"Per Capita {selected_product} Consumption (Barrels per Person per Year)",
                color="Per_Capita",
                color_continuous_scale="Viridis",
                text_auto='.2f'
            )
            
            fig.update_layout(
                xaxis_title="Country",
                yaxis_title="Barrels per Person per Year",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Per capita insights
            if not per_capita_data.empty:
                highest_per_capita = per_capita_data.iloc[0]
                lowest_per_capita = per_capita_data.iloc[-1]
                
                st.markdown(f"""
                ### Per Capita Consumption Insights
                
                **Highest Per Capita Consumer**: {highest_per_capita['CountryName']} ({highest_per_capita['Per_Capita']:.2f} barrels per person per year)
                
                **Lowest Per Capita Consumer**: {lowest_per_capita['CountryName']} ({lowest_per_capita['Per_Capita']:.2f} barrels per person per year)
                
                The difference between highest and lowest is {highest_per_capita['Per_Capita']/lowest_per_capita['Per_Capita']:.1f}x, 
                indicating vastly different consumption patterns.
                
                High per capita consumption may indicate {"energy-intensive economies, colder climates, or transportation-heavy lifestyles" if selected_product in ["Crude Oil", "Gasoline", "Diesel"] else "industrial focus or specific economic characteristics"}.
                """)
