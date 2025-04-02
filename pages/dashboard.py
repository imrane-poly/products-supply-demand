import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def show(data):
    """
    Improved Dashboard overview page
    
    Parameters:
    -----------
    data : pandas DataFrame
        Processed data
    """
    st.title("Oil Market Dashboard")
    st.write("A global overview of oil supply and demand patterns.")
    
    if data.empty:
        st.error("No data available. Please check data loading and filtering.")
        return
    
    # Extract unique values for filters
    products = sorted(data["Product"].unique())
    
    # Define major oil countries for easy selection
    major_countries = [
        "United States", "China", "Russia", "Saudi Arabia", 
        "India", "Japan", "Canada", "Germany", "Brazil", 
        "South Korea", "Iran", "United Kingdom"
    ]
    
    # Filter to only include countries that exist in the dataset
    available_countries = set(data["CountryName"].unique())
    major_countries = [country for country in major_countries if country in available_countries]
    
    # Define common flow types
    common_flows = [
        "Production", "Consumption", "Imports", "Exports", 
        "Stock Change", "Refinery Input"
    ]
    
    # Filter to only include flows that exist in the dataset
    available_flows = set(data["FlowBreakdown"].unique())
    common_flows = [flow for flow in common_flows if flow in available_flows]
    
    # Layout the filters in a more compact way
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Product selection with a default
        default_product_idx = 0
        if "Crude Oil" in products:
            default_product_idx = products.index("Crude Oil")
        elif "CRUDEOIL" in products:
            default_product_idx = products.index("CRUDEOIL")
        
        selected_product = st.selectbox(
            "Product", 
            products,
            index=default_product_idx
        )
    
    with col2:
        # Allow selection of multiple countries with major ones pre-selected
        default_countries = ["United States", "China", "Russia", "Saudi Arabia"]
        # Filter to only include default countries that exist in the dataset
        default_countries = [c for c in default_countries if c in available_countries]
        
        if not default_countries and major_countries:
            default_countries = [major_countries[0]]  # Select at least one country
            
        selected_countries = st.multiselect(
            "Countries",
            sorted(available_countries),
            default=default_countries
        )
        
        # If no countries selected, select all major countries
        if not selected_countries:
            selected_countries = major_countries
    
    with col3:
        # Date range selector
        end_date = data["ReferenceDate"].max().date()
        start_date = end_date - timedelta(days=365)  # Default to last year
        
        date_range = st.date_input(
            "Date Range",
            value=(start_date, end_date),
            min_value=data["ReferenceDate"].min().date(),
            max_value=end_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            st.info(f"No data available for {selected_flow}. Please select a different flow type.")

            start_date = data["ReferenceDate"].min().date()
    
    # Filter data based on selections
    filtered_data = data[
        (data["ReferenceDate"].dt.date >= start_date) &
        (data["ReferenceDate"].dt.date <= end_date) &
        (data["Product"] == selected_product) &
        (data["CountryName"].isin(selected_countries))
    ]
    
    if filtered_data.empty:
        st.warning(f"No data available for the selected filters. Please try different selections.")
        return
    
    # Show key stats in a header area
    st.markdown("---")
    st.subheader(f"{selected_product} Market Overview")
    
    # Create metrics row
    try:
        # Get latest date with data
        latest_date = filtered_data["ReferenceDate"].max()
        latest_data = filtered_data[filtered_data["ReferenceDate"] == latest_date]
        
        # Try to get production and consumption numbers
        production_flow = next((f for f in available_flows if "Production" in f), None)
        consumption_flow = next((f for f in available_flows if "Consumption" in f), None)
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            if production_flow:
                production = latest_data[latest_data["FlowBreakdown"] == production_flow]["ObservedValue"].sum()
                st.metric("Total Production", f"{production:,.0f} KB/D")
            else:
                st.metric("Total Countries", f"{len(selected_countries)}")
                
        with metrics_col2:
            if consumption_flow:
                consumption = latest_data[latest_data["FlowBreakdown"] == consumption_flow]["ObservedValue"].sum()
                st.metric("Total Consumption", f"{consumption:,.0f} KB/D")
            else:
                st.metric("Total Datapoints", f"{len(filtered_data):,}")
                
        with metrics_col3:
            st.metric("Date Range", f"{start_date} to {end_date}")
            
        with metrics_col4:
            # Calculate average value
            avg_value = filtered_data["ObservedValue"].mean()
            st.metric("Avg Value", f"{avg_value:,.0f} KB/D")
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
    
    # Create tab structure for detailed views
    st.markdown("---")
    tabs = st.tabs([
        "Global Trends", 
        "Country Comparison", 
        "Flow Analysis", 
        "Time Series"
    ])
    
    # Tab 1: Global Trends
    with tabs[0]:
        st.subheader("Global Trends")
        
        # Get production and consumption flow types if they exist
        production_flows = [f for f in available_flows if "Production" in f]
        consumption_flows = [f for f in available_flows if "Consumption" in f]
        
        # Time series chart by flow type
        flow_data = []
        available_flows = []
        
        # Add Production flows if they exist
        if production_flows:
            production_data = filtered_data[filtered_data["FlowBreakdown"].isin(production_flows)]
            if not production_data.empty:
                production_by_date = production_data.groupby("ReferenceDate")["ObservedValue"].sum().reset_index()
                production_by_date["Flow Type"] = "Production"
                flow_data.append(production_by_date)
                available_flows.append("Production")
        
        # Add Consumption flows if they exist
        if consumption_flows:
            consumption_data = filtered_data[filtered_data["FlowBreakdown"].isin(consumption_flows)]
            if not consumption_data.empty:
                consumption_by_date = consumption_data.groupby("ReferenceDate")["ObservedValue"].sum().reset_index()
                consumption_by_date["Flow Type"] = "Consumption"
                flow_data.append(consumption_by_date)
                available_flows.append("Consumption")
        
        # Add other important flows
        for flow_name in ["Imports", "Exports", "Stock Change"]:
            flow_patterns = [f for f in available_flows if flow_name in f]
            if flow_patterns:
                flow_specific_data = filtered_data[filtered_data["FlowBreakdown"].isin(flow_patterns)]
                if not flow_specific_data.empty:
                    flow_by_date = flow_specific_data.groupby("ReferenceDate")["ObservedValue"].sum().reset_index()
                    flow_by_date["Flow Type"] = flow_name
                    flow_data.append(flow_by_date)
                    available_flows.append(flow_name)
        
        if flow_data:
            # Combine all flow data
            combined_flows = pd.concat(flow_data)
            
            # Create time series chart
            fig = px.line(
                combined_flows,
                x="ReferenceDate",
                y="ObservedValue",
                color="Flow Type",
                title=f"Global {selected_product} Trends",
                labels={"ObservedValue": "Volume (KB/D)", "ReferenceDate": "Date"}
            )
            
            fig.update_layout(
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
            
            # Add balance chart if we have both production and consumption
            if "Production" in available_flows and "Consumption" in available_flows:
                # Create production and consumption data
                prod_data = combined_flows[combined_flows["Flow Type"] == "Production"]
                cons_data = combined_flows[combined_flows["Flow Type"] == "Consumption"]
                
                # Merge production and consumption data
                balance_data = pd.merge(
                    prod_data,
                    cons_data,
                    on="ReferenceDate",
                    suffixes=("_production", "_consumption")
                )
                
                # Calculate balance
                balance_data["Balance"] = balance_data["ObservedValue_production"] - balance_data["ObservedValue_consumption"]
                
                # Create balance chart
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add production and consumption as bars
                fig2.add_trace(
                    go.Bar(
                        x=balance_data["ReferenceDate"],
                        y=balance_data["ObservedValue_production"],
                        name="Production",
                        marker_color="green",
                        opacity=0.7
                    ),
                    secondary_y=False
                )
                
                fig2.add_trace(
                    go.Bar(
                        x=balance_data["ReferenceDate"],
                        y=balance_data["ObservedValue_consumption"],
                        name="Consumption",
                        marker_color="red",
                        opacity=0.7
                    ),
                    secondary_y=False
                )
                
                # Add balance line
                fig2.add_trace(
                    go.Scatter(
                        x=balance_data["ReferenceDate"],
                        y=balance_data["Balance"],
                        name="Balance",
                        line=dict(color="blue", width=2)
                    ),
                    secondary_y=True
                )
                
                # Add zero line for balance
                fig2.add_shape(
                    type="line",
                    x0=balance_data["ReferenceDate"].min(),
                    y0=0,
                    x1=balance_data["ReferenceDate"].max(),
                    y1=0,
                    line=dict(color="gray", dash="dash"),
                    yref="y2"
                )
                
                # Update layout
                fig2.update_layout(
                    title=f"{selected_product} Supply & Demand Balance",
                    barmode="group",
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                fig2.update_yaxes(title_text="Volume (KB/D)", secondary_y=False)
                fig2.update_yaxes(title_text="Balance (KB/D)", secondary_y=True)
                fig2.update_xaxes(title_text="Date")
                
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No flow data available for the selected filters.")
    
    # Tab 2: Country Comparison
    with tabs[1]:
        st.subheader("Country Comparison")
        
        # Choose a flow breakdown to compare
        available_flow_options = sorted(filtered_data["FlowBreakdown"].unique())
        
        # Try to default to Production or Consumption
        default_flow_idx = 0
        for flow_type in ["Production", "Consumption"]:
            matching_flows = [i for i, f in enumerate(available_flow_options) if flow_type in f]
            if matching_flows:
                default_flow_idx = matching_flows[0]
                break
        
        selected_flow = st.selectbox(
            "Select Flow Type",
            available_flow_options,
            index=default_flow_idx
        )
        
        # Filter data for the selected flow
        flow_data = filtered_data[filtered_data["FlowBreakdown"] == selected_flow]
        
        if not flow_data.empty:
            # Get latest date data
            latest_date = flow_data["ReferenceDate"].max()
            latest_flow_data = flow_data[flow_data["ReferenceDate"] == latest_date]
            
            # Aggregate by country
            country_totals = latest_flow_data.groupby("CountryName")["ObservedValue"].sum().reset_index()
            
            # Sort by value
            country_totals = country_totals.sort_values("ObservedValue", ascending=False)
            
            # Create bar chart
            fig = px.bar(
                country_totals,
                x="CountryName",
                y="ObservedValue",
                title=f"Country Comparison: {selected_flow} of {selected_product}",
                color="ObservedValue",
                color_continuous_scale="Viridis",
                labels={"ObservedValue": "Volume (KB/D)", "CountryName": "Country"},
                text_auto='.2s'
            )
            
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Time series comparison by country
            st.subheader("Time Series by Country")
            
            # Group by date and country
            country_time_series = flow_data.groupby(["ReferenceDate", "CountryName"])["ObservedValue"].sum().reset_index()
            
            # Create line chart
            fig2 = px.line(
                country_time_series,
                x="ReferenceDate",
                y="ObservedValue",
                color="CountryName",
                title=f"Time Series: {selected_flow} of {selected_product} by Country",
                labels={"ObservedValue": "Volume (KB/D)", "ReferenceDate": "Date"}
            )
            
            fig2.update_layout(
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Show market share pie chart
            st.subheader("Market Share Analysis")
            
            # Calculate total
            total_value = country_totals["ObservedValue"].sum()
            
            # Calculate percentage
            country_totals["Percentage"] = (country_totals["ObservedValue"] / total_value * 100).round(1)
            
            # Get top 8 countries and group others
            if len(country_totals) > 8:
                top_countries = country_totals.head(8)
                other_countries = country_totals.iloc[8:]
                
                other_row = pd.DataFrame({
                    "CountryName": ["Others"],
                    "ObservedValue": [other_countries["ObservedValue"].sum()],
                    "Percentage": [other_countries["Percentage"].sum()]
                })
                
                pie_data = pd.concat([top_countries, other_row])
            else:
                pie_data = country_totals
            
            # Create pie chart
            fig3 = px.pie(
                pie_data,
                values="ObservedValue",
                names="CountryName",
                title=f"Market Share: {selected_flow} of {selected_product}",
                hover_data=["Percentage"],
                labels={"Percentage": "Market Share (%)"}
            )
            
            fig3.update_traces(textposition='inside', textinfo='percent+label')
            fig3.update_layout(height=500)
            
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No flow data available for the selected filters.")

                
    # Tab 3: Flow Analysis
    with tabs[2]:
        st.subheader("Flow Analysis")
        
        # Find all available flow breakdowns and categorize them
        flow_categories = {}
        
        for flow in filtered_data["FlowBreakdown"].unique():
            # Try to categorize flows
            if "Production" in flow:
                category = "Production"
            elif "Consumption" in flow:
                category = "Consumption"
            elif "Import" in flow:
                category = "Imports"
            elif "Export" in flow:
                category = "Exports"
            elif "Stock" in flow:
                category = "Stocks"
            elif "Refinery" in flow:
                category = "Refinery"
            else:
                category = "Other"
            
            if category not in flow_categories:
                flow_categories[category] = []
            
            flow_categories[category].append(flow)
        
        # Select a country for detailed flow analysis
        selected_country = st.selectbox(
            "Select Country for Flow Analysis",
            sorted(filtered_data["CountryName"].unique())
        )
        
        # Filter data for the selected country
        country_data = filtered_data[filtered_data["CountryName"] == selected_country]
        
        if not country_data.empty:
            # Get the latest date
            latest_date = country_data["ReferenceDate"].max()
            
            # Get data for all available flows for this country
            latest_flows = []
            
            for flow in country_data["FlowBreakdown"].unique():
                flow_value = country_data[
                    (country_data["FlowBreakdown"] == flow) & 
                    (country_data["ReferenceDate"] == latest_date)
                ]["ObservedValue"].sum()
                
                if flow_value > 0:
                    # Find the category for this flow
                    category = next((cat for cat, flows in flow_categories.items() if flow in flows), "Other")
                    
                    latest_flows.append({
                        "Flow": flow,
                        "Category": category,
                        "Value": flow_value
                    })
            
            if latest_flows:
                # Convert to DataFrame
                flows_df = pd.DataFrame(latest_flows)
                
                # Sort by value
                flows_df = flows_df.sort_values("Value", ascending=False)
                
                # Create grouped bar chart
                fig = px.bar(
                    flows_df,
                    x="Flow",
                    y="Value",
                    color="Category",
                    title=f"Flow Analysis for {selected_country}: {selected_product}",
                    labels={"Value": "Volume (KB/D)", "Flow": "Flow Type"},
                    text_auto='.2s'
                )
                
                fig.update_layout(height=500)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Flow time series analysis
                st.subheader("Flow Time Series Analysis")
                
                # Select top flows for time series visualization
                top_flows = flows_df.head(5)["Flow"].tolist()
                
                # Allow user to select which flows to visualize
                selected_flows = st.multiselect(
                    "Select Flows to Visualize",
                    sorted(country_data["FlowBreakdown"].unique()),
                    default=top_flows[:3]  # Default to top 3 flows
                )
                
                if selected_flows:
                    # Filter data for selected flows
                    flow_time_series = country_data[country_data["FlowBreakdown"].isin(selected_flows)]
                    
                    # Group by date and flow
                    flow_by_date = flow_time_series.groupby(["ReferenceDate", "FlowBreakdown"])["ObservedValue"].sum().reset_index()
                    
                    # Create line chart
                    fig2 = px.line(
                        flow_by_date,
                        x="ReferenceDate",
                        y="ObservedValue",
                        color="FlowBreakdown",
                        title=f"Time Series: {selected_country} {selected_product} Flows",
                        labels={"ObservedValue": "Volume (KB/D)", "ReferenceDate": "Date", "FlowBreakdown": "Flow Type"}
                    )
                    
                    fig2.update_layout(
                        height=500,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Please select at least one flow to visualize.")
            else:
                st.info(f"No flow data available for {selected_country}.")
        else:
            st.info(f"No data available for {selected_country}.")
    
    # Tab 4: Time Series
    with tabs[3]:
        st.subheader("Time Series Analysis")
        
        # Allow selection of flow type
        selected_ts_flow = st.selectbox(
            "Select Flow Type for Time Series",
            sorted(filtered_data["FlowBreakdown"].unique()),
            index=0
        )
        
        # Filter data for selected flow
        ts_data = filtered_data[filtered_data["FlowBreakdown"] == selected_ts_flow]
        
        if not ts_data.empty:
            # Aggregate by date
            time_series = ts_data.groupby("ReferenceDate")["ObservedValue"].sum().reset_index()
            
            # Extract year and month for trend analysis
            time_series["Year"] = time_series["ReferenceDate"].dt.year
            time_series["Month"] = time_series["ReferenceDate"].dt.month
            
            # Create interactive time series
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=time_series["ReferenceDate"],
                    y=time_series["ObservedValue"],
                    mode="lines+markers",
                    name=f"Total {selected_ts_flow}",
                    line=dict(color="blue", width=2)
                )
            )
            
            # Calculate and add trend line
            x = np.arange(len(time_series))
            y = time_series["ObservedValue"].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            fig.add_trace(
                go.Scatter(
                    x=time_series["ReferenceDate"],
                    y=p(x),
                    mode="lines",
                    name="Trend",
                    line=dict(color="red", width=2, dash="dash")
                )
            )
            
            # Update layout
            fig.update_layout(
                title=f"Time Series: {selected_ts_flow} of {selected_product}",
                xaxis_title="Date",
                yaxis_title="Volume (KB/D)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly pattern analysis
            st.subheader("Monthly Pattern Analysis")
            
            # Calculate monthly averages
            monthly_avg = time_series.groupby("Month")["ObservedValue"].mean().reset_index()
            
            # Map month numbers to names
            month_names = {
                1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
            }
            monthly_avg["Month_Name"] = monthly_avg["Month"].map(month_names)
            
            # Create bar chart
            fig2 = px.bar(
                monthly_avg,
                x="Month_Name",
                y="ObservedValue",
                title=f"Monthly Pattern: {selected_ts_flow} of {selected_product}",
                color="ObservedValue",
                color_continuous_scale="Viridis",
                text_auto='.2s',
                labels={"ObservedValue": "Average Volume (KB/D)", "Month_Name": "Month"}
            )
            
            fig2.update_layout(
                height=400,
                xaxis=dict(
                    categoryorder='array',
                    categoryarray=list(month_names.values())
                )
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Year-over-year analysis
            st.subheader("Year-over-Year Analysis")
            
            # Calculate annual averages
            yearly_avg = time_series.groupby("Year")["ObservedValue"].mean().reset_index()
            
            # Create bar chart
            fig3 = px.bar(
                yearly_avg,
                x="Year",
                y="ObservedValue",
                title=f"Annual Average: {selected_ts_flow} of {selected_product}",
                color="ObservedValue",
                color_continuous_scale="Viridis",
                text_auto='.2s',
                labels={"ObservedValue": "Average Volume (KB/D)", "Year": "Year"}
            )
            
            fig3.update_layout(height=400)
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Growth rate calculation
            if len(yearly_avg) > 1:
                yearly_avg["Growth"] = yearly_avg["ObservedValue"].pct_change() * 100
                
                # Create growth rate chart
                fig4 = px.bar(
                    yearly_avg.dropna(),
                    x="Year",
                    y="Growth",
                    title=f"Annual Growth Rate: {selected_ts_flow} of {selected_product}",
                    text_auto='.1f',
                    labels={"Growth": "Growth Rate (%)", "Year": "Year"},
                    color="Growth",
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0
                )
                
                # Add zero line
                fig4.add_shape(
                    type="line",
                    x0=yearly_avg["Year"].min(),
                    y0=0,
                    x1=yearly_avg["Year"].max(),
                    y1=0,
                    line=dict(color="gray", dash="dash")
                )
                
                fig4.update_layout(height=400)
                
                st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info(f"No data available for {selected_ts_flow}.")
    
    # Add footer with data summary
    st.markdown("---")
    st.caption(f"Data summary: {len(filtered_data):,} records for {len(selected_countries)} countries from {start_date} to {end_date}")
