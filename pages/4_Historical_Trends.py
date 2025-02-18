import streamlit as st
import pandas as pd
import plotly.express as px


# Load the dataset
def load_data():
    df = pd.read_csv("/Users/sarvagya/Developer/agaDatathon2/compiled_stats.csv")
    df["year"] = df["year"].astype(int)
    df["total_outlayed_amount"] = pd.to_numeric(df["total_outlayed_amount"], errors='coerce')
    return df


dataset = load_data()


# Compute normalized funding
def normalize_funding(df):
    first_year = df[df["year"] == 2020]["total_outlayed_amount"].mean()
    df["normalized_funding"] = df["total_outlayed_amount"] / first_year if first_year else df["total_outlayed_amount"]
    return df


dataset = dataset.groupby(["awarding_agency_name", "year"], as_index=False)["total_outlayed_amount"].mean()
dataset = dataset.groupby("awarding_agency_name").apply(normalize_funding).reset_index(drop=True)

# Streamlit UI
st.set_page_config(page_title="Historical Trends", page_icon="📉", layout="wide")
st.title("📉 Historical Funding Trends")
st.markdown("Analyze how funding has changed over time for different agencies.")

# Agency Dropdown
agency_input = st.selectbox("**Select an awarding agency:**", dataset["awarding_agency_name"].unique(),
                            key="agency_select")


def plot_subagency_funding_trend(agency_name, data):
    agency_data = data[data["awarding_agency_name"] == agency_name]

    if agency_data.empty:
        st.warning(f"No data available for {agency_name}.")
        return

    fig = px.line(
        agency_data,
        x="year",
        y="normalized_funding",
        markers=True,
        title=f"Funding Trend for {agency_name} (Normalized to 2020)",
        labels={"normalized_funding": "Funding Change (Normalized to 2020)", "year": "Year"},
        line_shape="linear"
    )
    fig.add_hline(y=1, line_dash="dash", line_color="black", annotation_text="Baseline (2020)")

    st.plotly_chart(fig, use_container_width=True)


plot_subagency_funding_trend(agency_input, dataset)