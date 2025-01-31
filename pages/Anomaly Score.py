import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import gdown
import os

# Set up Streamlit page
st.set_page_config(
    page_title="Anomaly Score Dashboard",
    page_icon="⚠️",
    layout="wide"
)

# Apply custom padding
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .stMetric {
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "dataset" in st.session_state and st.session_state["dataset_loaded"]:
    dataset = st.session_state["dataset"]
else:
    st.warning("Dataset is still loading, please wait...")
    st.stop()

# Function to calculate SMB Transparency Score
def calculate_smb_transparency_score(df, agency_name):
    """
    Evaluates an agency’s funding transparency for SMBs.
    Adjusts penalty weights based on anomaly severity with a softer scoring approach.

    Parameters:
        df (DataFrame): Federal spending dataset.
        agency_name (str): The name of the awarding agency.

    Returns:
        float: Transparency score (40-100).
        float: Anomaly-to-record ratio.
    """
    agency_df = df[df["awarding_agency_name"] == agency_name]

    if agency_df.empty:
        return None, None

    total_records = len(agency_df)

    # Define anomaly categories with softer weights
    anomalies = {
        "Duplicate Awards": (agency_df.duplicated(subset=["assistance_award_unique_key", "award_id_fain", "award_id_uri"]).sum(), 0.3),
        "Negative/Extreme Values": (((agency_df["total_obligated_amount"] < 0) | (agency_df["total_obligated_amount"] > 1_000_000_000)).sum(), 0.3),
        "Timing Issues": ((agency_df["award_base_action_date"] > agency_df["award_latest_action_date"]).sum(), 0.3),
        "Funding Mismatch": ((agency_df["total_funding_amount"] < agency_df["total_obligated_amount"]).sum(), 0.3),
        "Payment Accuracy Issues": ((agency_df["total_outlayed_amount"] > agency_df["total_obligated_amount"]).sum(), 0.3)
    }

    total_weighted_anomalies = sum(count * weight for count, weight in anomalies.values())
    anomaly_ratio = total_weighted_anomalies / total_records if total_records > 0 else 0

    # Adjust transparency score with gentler logarithmic scaling
    base_score = 100
    score_penalty = min(60, np.log1p(total_weighted_anomalies) * 8) if total_weighted_anomalies > 0 else 0
    transparency_score = max(40, base_score - score_penalty)  # Minimum score of 40

    # Small bonus for agencies with <3% anomalies
    if anomaly_ratio < 0.03:
        transparency_score = min(100, transparency_score + 5)

    return transparency_score, anomaly_ratio

# Calculate anomaly scores for all agencies
st.title("⚠️ Anomaly Score Dashboard")
st.markdown("This dashboard evaluates and visualizes funding anomalies across awarding agencies.")

# Apply function to all unique agencies
unique_agencies = dataset["awarding_agency_name"].unique()
agency_scores = []

for agency in unique_agencies:
    score, anomaly_ratio = calculate_smb_transparency_score(dataset, agency)
    if score is not None:
        agency_scores.append({"Agency": agency, "Transparency Score": score, "Anomaly Ratio": anomaly_ratio})

# Convert results to DataFrame
score_df = pd.DataFrame(agency_scores)

# Display scores
st.subheader("📊 Agency Transparency Scores")
st.write(score_df.sort_values(by="Transparency Score", ascending=False))

# **Visualization: Agencies by Anomaly Score**
st.subheader("📉 Transparency Score by Agency")

fig = px.bar(
    score_df.sort_values(by="Transparency Score", ascending=True),
    x="Transparency Score",
    y="Agency",
    orientation="h",
    title="Transparency Score of Awarding Agencies",
    labels={"Transparency Score": "Transparency Score (Higher is Better)", "Agency": "Awarding Agency"},
    color="Transparency Score",
    color_continuous_scale="RdYlGn"
)

st.plotly_chart(fig, use_container_width=True)
