import streamlit as st
import pandas as pd
import inflect
import plotly.express as px
import plotly.graph_objects as go
import gdown

# Initialize inflect engine for number-to-word conversion
p = inflect.engine()

st.set_page_config(
    page_title="Federal Spending Transparency Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 4.5rem;
        padding-bottom: 4rem;
        padding-left: 8rem;
        padding-right: 8rem;
    }
    .stMetric {
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the dataset
#data_path = '/Users/sarvagya/Developer/agaDatathon2/grant_combined.csv'


if "dataset" in st.session_state and st.session_state["dataset_loaded"]:
    dataset = st.session_state["dataset"]
else:
    st.warning("Dataset is still loading, please wait...")
    st.stop()

# Calculate Metrics
total_grants = dataset["assistance_award_unique_key"].nunique()
total_agencies = dataset["awarding_agency_name"].nunique()
total_recipients = dataset["recipient_name"].nunique()  # New metric

# Hardcoded Date Range
date_range_start = "2023-01-01"
date_range_end = "2023-12-31"


# Convert large numbers into words (abbreviations)
def number_to_abbreviation(amount):
    if amount >= 1_000_000_000:  # Billions
        return f"{amount / 1_000_000_000:.2f}B"
    elif amount >= 1_000_000:  # Millions
        return f"{amount / 1_000_000:.2f}M"
    elif amount >= 1_000:  # Thousands
        return f"{amount / 1_000:.2f}K"
    else:
        return f"{amount:,.2f}"


# Get numeric values
total_obligated_amount = dataset["total_obligated_amount"].sum()
total_outlayed_amount = dataset["total_outlayed_amount"].sum()

# Convert to abbreviated format
total_obligated_abbr = number_to_abbreviation(total_obligated_amount)
total_outlayed_abbr = number_to_abbreviation(total_outlayed_amount)

# Page Title
st.title("📊 Grants Data Dashboard")
st.markdown("A detailed interactive dashboard to explore US government grant data.")

# **Key Metrics Section**
st.subheader("📌 Key Metrics")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Obligated Amount", f"${total_obligated_abbr}")
    st.metric("Total Grants Analyzed", f"{total_grants:,}")

with col2:
    st.metric("Total Outlayed Amount", f"${total_outlayed_abbr}")
    st.metric("Total Agencies Analyzed", f"{total_agencies:,}")

# **Additional Metric: Total Recipients**
st.metric("Total Recipients Analyzed", f"{total_recipients:,}")

# **Date Range**
st.subheader("📅 Date Range of Grants Data")
st.markdown(f"📆 **From:** {date_range_start} **to:** {date_range_end}")

st.subheader("📊 Anomaly Visualizer")

# Vlad's Viz Supporter
obligated_negatives = dataset[dataset["total_obligated_amount"] < 0]
obligated_extremes = dataset[dataset["total_obligated_amount"] > 1_000_000_000]
outlay_negatives = dataset[(dataset["total_outlayed_amount"] < 0)]
outlay_extremes = dataset[(dataset["total_outlayed_amount"] > 1_000_000_000)]
funding_negatives = dataset[dataset["total_funding_amount"] < 0]
funding_extremes = dataset[dataset["total_funding_amount"] > 1_000_000_000]
obligated_negatives["recipient_state_name"].value_counts()
obligated_extremes['awarding_sub_agency_name'].value_counts()
outlay_negatives['awarding_sub_agency_name'].value_counts()
outlay_extremes['awarding_sub_agency_name'].value_counts()
funding_negatives['awarding_sub_agency_name'].value_counts()
funding_extremes['awarding_sub_agency_name'].value_counts()

#Viz 1
def plot_subagency_funding_histogram(agency_name, data):
    """
    Creates an interactive Plotly bar chart showing the deviation of
    sub-agency funding from the agency's average funding.

    Parameters:
        agency_name (str): Name of the awarding agency to filter.
        data (DataFrame): The dataframe containing sub-agency funding details.
    """

    # Filter for the selected agency
    agency_data = data[data["awarding_agency_name"] == agency_name]

    if agency_data.empty:
        st.warning(f"No sub-agency data found for '{agency_name}'.")
        return

    # Group by sub-agency and calculate the average funding
    subagency_funding = agency_data.groupby("awarding_sub_agency_name")["total_funding_amount"].mean().reset_index()
    subagency_funding.rename(columns={"total_funding_amount": "Average Funding Given"}, inplace=True)

    # Calculate global average funding for the agency
    global_avg_funding = subagency_funding["Average Funding Given"].mean()

    # Calculate deviations from the global agency average
    subagency_funding["Funding Deviation"] = subagency_funding["Average Funding Given"] - global_avg_funding

    # Sort sub-agencies by funding deviation
    subagency_funding = subagency_funding.sort_values(by="Funding Deviation", ascending=False)

    # Create color scale based on deviation
    colors = subagency_funding["Funding Deviation"].apply(lambda x: "green" if x >= 0 else "red")

    # Create interactive bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=subagency_funding["awarding_sub_agency_name"],
        y=subagency_funding["Funding Deviation"],
        text=[f"${val:,.0f}" for val in subagency_funding["Average Funding Given"]],  # Display values
        textposition="outside",
        marker=dict(color=colors),
        name="Deviation from Avg Funding"
    ))

    # Add a horizontal line for the global average funding
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0,
        y1=0,
        xref="paper",
        yref="y",
        line=dict(color="black", width=2, dash="dash"),
    )

    # Layout & Styling
    fig.update_layout(
        title=f"📊 Funding Deviation of Sub-Agencies in {agency_name}",
        xaxis_title="Sub-Agency",
        yaxis_title="Deviation from Agency Average",
        xaxis_tickangle=-45,
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=80),
        height=600
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# Example usage inside Streamlit
agency_input = st.selectbox("Select an awarding agency:", dataset["awarding_agency_name"].unique(), key="agency_select")
plot_subagency_funding_histogram(agency_input, dataset)
