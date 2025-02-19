import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("master_agency_subagency_funding_data.csv")

df = load_data()

# Convert date and extract year
df["year"] = df["year"].astype(int)

# Streamlit App
st.title("📈 Historical Funding Trends")
st.markdown(
    "<hr style='border: 0.5px solid lightgray; margin-top: -5px;'>", 
    unsafe_allow_html=True
)

# Get unique agencies list
agency_list = df["awarding_agency_name"].unique()

# Set default agency (check if it exists in the dataset)
default_agency = "Department of Housing and Urban Development"
default_index = list(agency_list).index(default_agency) if default_agency in agency_list else 0  # Default to index 0 if not found

# Agency Selection Dropdown
agency_input = st.selectbox(
    "**Select an awarding agency:**",
    agency_list,
    index=default_index,  # Set default index
    key="agency_select"
)

# Filter for the selected agency
agency_data = df[df["awarding_agency_name"] == agency_input]

if agency_data.empty:
    st.warning(f"No data available for '{agency_input}'.")
else:
    # Select Top 5 Sub-Agencies by total outlayed amount
    top_5_subagencies = (
        agency_data.groupby("awarding_sub_agency_name")["total_outlayed_amount"]
        .sum()
        .nlargest(5)
        .index
    )

    agency_data = agency_data[agency_data["awarding_sub_agency_name"].isin(top_5_subagencies)]
    agency_data["year"] = agency_data["year"].astype(str)  # Convert to string 

    # Plotly Line Chart for Funding Trend
    fig = px.line(
        agency_data,
        x="year",
        y="normalized_funding",
        color="awarding_sub_agency_name",
        markers=True,
        title=f"Funding Trend for Top 5 Sub-Agencies under {agency_input}",
        labels={
        "normalized_funding": "Funding Change (Normalized to 2020)",
        "year": "Year",
        "awarding_sub_agency_name": "Sub-Agency"  
    },
)

    # Add baseline line
    fig.add_hline(y=1, line_dash="dash", line_color="black", annotation_text="Baseline (2020)")
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)

# **Top 5 Sub-Agencies by YoY Growth for Selected Agency**

# Filter for the selected agency
agency_data = df[df["awarding_agency_name"] == agency_input]

# Compute YoY growth for each sub-agency
agency_data = agency_data.sort_values(by=["awarding_sub_agency_name", "year"])
agency_data["prev_year_amount"] = agency_data.groupby("awarding_sub_agency_name")["total_outlayed_amount"].shift(1)

# Calculate YoY Growth %
agency_data["yoy_growth"] = (agency_data["total_outlayed_amount"] - agency_data["prev_year_amount"]) / agency_data["prev_year_amount"]

# Select Top 5 Sub-Agencies by **average YoY growth**
top_5_subagencies_growth = (
    agency_data.groupby("awarding_sub_agency_name")["yoy_growth"]
    .mean()
    .nlargest(5)
    .index
)

# Filter for only top 5 sub-agencies
agency_data = agency_data[agency_data["awarding_sub_agency_name"].isin(top_5_subagencies_growth)]
agency_data["year"] = agency_data["year"].astype(str)

# Plot YoY Growth as Line Chart
fig_yoy = px.line(
    agency_data,
    x="year",
    y="yoy_growth",
    color="awarding_sub_agency_name",
    markers=True,
    title=f"📈 YoY Growth in Outlayed Amount - Top 5 Sub-Agencies under {agency_input}",
    labels={
        "yoy_growth": "YoY Growth (%)",
        "year": "Year",
        "awarding_sub_agency_name": "Sub-Agency"  
    },
 )

# Format YoY Growth to Percentage
fig_yoy.update_yaxes(tickformat=".1%")
fig_yoy.update_xaxes(type="category")
st.plotly_chart(fig_yoy, use_container_width=True)