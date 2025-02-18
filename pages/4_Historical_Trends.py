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

# Agency Selection Dropdown
agency_input = st.selectbox(
    "**Select an awarding agency:**",
    df["awarding_agency_name"].unique(),
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

    # Plotly Line Chart for Funding Trend
    fig = px.line(
        agency_data,
        x="year",
        y="normalized_funding",
        color="awarding_sub_agency_name",
        markers=True,
        title=f"Funding Trend for Top 5 Sub-Agencies under {agency_input}",
        labels={"normalized_funding": "Funding Change (Normalized to 2020)", "year": "Year"},
    )

    # Add baseline line
    fig.add_hline(y=1, line_dash="dash", line_color="black", annotation_text="Baseline (2020)")

    st.plotly_chart(fig, use_container_width=True)