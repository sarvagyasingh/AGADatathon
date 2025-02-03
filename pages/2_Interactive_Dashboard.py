import streamlit as st
import pandas as pd
import inflect
import plotly.express as px
import plotly.graph_objects as go

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
# data_path = '/Users/sarvagya/Developer/agaDatathon2/grant_combined.csv'


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
# def number_to_abbreviation(amount):
#     if amount >= 1_000_000_000:  # Billions
#         return f"{amount / 1_000_000_000:.2f}B"
#     elif amount >= 1_000_000:  # Millions
#         return f"{amount / 1_000_000:.2f}M"
#     elif amount >= 1_000:  # Thousands
#         return f"{amount / 1_000:.2f}K"
#     else:
#         return f"{amount:,.2f}"

def number_to_abbreviation(amount):
    print(amount)
    amount = float(amount)  # Ensure amount is numeric

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

# Top 5 agencie
top_5_agencies = dataset.groupby('awarding_agency_name')['total_outlayed_amount'].sum().nlargest(5)
st.subheader("Top 5 Agencies by Total Outlayed Amount")
top_5_agencies

split_columns = dataset['prime_award_summary_place_of_performance_cd_original'].str.split('-', n=1, expand=True)
split_columns = split_columns.rename(
    columns={0: 'place_of_performance_state_code', 1: 'place_of_performance_state_code_not_needed'})
dataset = pd.concat([dataset, split_columns], axis=1)
dataset['obligation_utilization_ratio'] = (dataset['total_outlayed_amount'] / dataset['total_obligated_amount'])

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


# Viz 1
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
    print(f"Length {len(agency_data)}")

    if agency_data.empty:
        st.warning(f"No sub-agency data found for '{agency_name}'.")
        return

    total_counts = len(agency_data)

    # Compute statistics
    total_obligated = agency_data['total_obligated_amount'].sum()
    total_obligated_abbr = number_to_abbreviation(total_obligated)
    total_outlayed = agency_data['total_outlayed_amount'].sum()
    total_outlayed_abbr = number_to_abbreviation(total_outlayed)
    nan_obligated = agency_data["total_obligated_amount"].isna().sum() / total_counts
    nan_outlayed = agency_data["total_outlayed_amount"].isna().sum() / total_counts
    total_recipients_served = agency_data["recipient_name"].nunique()

    # Display stats in Streamlit
    st.markdown(f"### 📌 Key Metrics")

    col1, col2 = st.columns(2)

    col1.metric("Total Awards", f"{total_counts:,}")
    col2.metric("Unique Recipients Served", f"{total_recipients_served:,}")

    col3, col4 = st.columns(2)

    col3.metric("Total Obligated Amount", f"${total_obligated_abbr}")
    col4.metric("Total Outlayed Amount", f"${total_outlayed_abbr}")

    # Group by sub-agency and calculate the average funding
    subagency_funding = agency_data.groupby("awarding_sub_agency_name")["total_funding_amount"].mean().reset_index()
    subagency_funding.rename(columns={"total_funding_amount": "Average Funding Given"}, inplace=True)

    # Calculate global average funding for the agency
    global_avg_funding = subagency_funding["Average Funding Given"].mean()

    # Calculate deviations from the global agency average
    subagency_funding["Funding Deviation"] = subagency_funding["Average Funding Given"] - global_avg_funding

    # Sort sub-agencies by funding deviation
    subagency_funding = subagency_funding.sort_values(by="Funding Deviation", ascending=False)

    # **Separate positive and negative deviations**
    positive_deviation = subagency_funding[subagency_funding["Funding Deviation"] >= 0]
    negative_deviation = subagency_funding[subagency_funding["Funding Deviation"] < 0]

    # Create interactive bar chart
    fig = go.Figure()

    # **Green Bars (Above Avg)**
    fig.add_trace(go.Bar(
        x=positive_deviation["awarding_sub_agency_name"],
        y=positive_deviation["Funding Deviation"],
        text=[f"${val:,.0f}" for val in positive_deviation["Average Funding Given"]],
        textposition="outside",
        marker=dict(color="green"),
        name="Above Agency Average"  # ✅ Separate legend entry
    ))

    # **Red Bars (Below Avg)**
    fig.add_trace(go.Bar(
        x=negative_deviation["awarding_sub_agency_name"],
        y=negative_deviation["Funding Deviation"],
        text=[f"${val:,.0f}" for val in negative_deviation["Average Funding Given"]],
        textposition="outside",
        marker=dict(color="red"),
        name="Below Agency Average"  # ✅ Separate legend entry
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


def plot_agency_funding_by_state(df, agency_name):
    """
    Generates a choropleth map displaying the number of funding instances
    per state for a given agency in Streamlit.

    Parameters:
        df (DataFrame): Federal spending dataset.
        agency_name (str): The agency name to filter and visualize.

    Returns:
        Displays an interactive Plotly choropleth map in Streamlit.
    """

    # Filter for the selected agency
    agency_data = df[df["awarding_agency_name"] == agency_name]

    if agency_data.empty:
        st.warning(f"No records found for '{agency_name}'.")
        return

    # Group by state and count number of funding instances
    state_funding_counts = agency_data.groupby(
        ['recipient_state_name', 'recipient_state_code']
    ).size().reset_index(name="Funding Instances")

    # Create choropleth map
    fig = px.choropleth(
        state_funding_counts,
        locations="recipient_state_code",
        locationmode="USA-states",
        color="Funding Instances",
        hover_name="recipient_state_name",
        color_continuous_scale="Viridis",
        scope="usa",
        title=f"📍 Funding Instances Per State - {agency_name}"
    )

    # Adjust map appearance
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

    # Display the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def outlayed_amount_per_state(df, agency_name):
    agency_data = df[df["awarding_agency_name"] == agency_name]

    state_outlayed_amount = agency_data.groupby(
        ['place_of_performance_state_code', 'primary_place_of_performance_state_name']
    )["total_outlayed_amount"].sum().reset_index()

    # Create choropleth map
    fig = px.choropleth(
        state_outlayed_amount,
        locations='place_of_performance_state_code',
        locationmode='USA-states',
        color='total_outlayed_amount',
        hover_name='primary_place_of_performance_state_name',
        color_continuous_scale="Magma",
        scope="usa",
        title="📍 Total Outlayed Amount Per State ($)"
    )

    # Center the map properly
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},  # Adjust top margin for title
        geo=dict(center={"lat": 37.0902, "lon": -95.7129}, projection_scale=1)  # Center on USA
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)  # Makes it responsive and centered


outlayed_amount_per_state(dataset, agency_input)

# 🏛️ Use Existing Dropdown to Control This Visualization
plot_agency_funding_by_state(dataset, agency_input)


def plot_subagency_obligated_vs_outlayed(df, agency_name):
    """
    Generates a grouped bar chart comparing obligated vs. outlayed amounts
    for each sub-agency within the selected agency.

    Parameters:
        df (DataFrame): Federal spending dataset.
        agency_name (str): The selected agency name.

    Returns:
        Displays an interactive Plotly grouped bar chart in Streamlit.
    """

    # Filter data for the selected agency
    agency_data = df[df["awarding_agency_name"] == agency_name]

    if agency_data.empty:
        st.warning(f"No records found for '{agency_name}'.")
        return

    # Group by sub-agency and sum up obligated & outlayed amounts
    subagency_funding = agency_data.groupby("awarding_sub_agency_name")[
        ["total_obligated_amount", "total_outlayed_amount"]
    ].sum().reset_index()

    # Check if data is available
    if subagency_funding.empty:
        st.warning(f"No funding data available for sub-agencies under '{agency_name}'.")
        return

    top_10_sub_agencies = subagency_funding.nlargest(10, "total_obligated_amount")

    # Create grouped bar chart
    fig = px.bar(
        top_10_sub_agencies.melt(id_vars="awarding_sub_agency_name",
                                 value_vars=["total_obligated_amount", "total_outlayed_amount"],
                                 var_name="Funding Type",
                                 value_name="Amount"),
        x="awarding_sub_agency_name",
        y="Amount",
        color="Funding Type",
        barmode="group",
        title=f"💰 Obligated vs. Outlayed Amount by Sub-Agency - {agency_name}",
        labels={"awarding_sub_agency_name": "Sub-Agency", "Amount": "Funding Amount ($)"},
        text_auto=True
    )

    # Layout Adjustments
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(
        xaxis_title="Sub-Agency",
        yaxis_title="Funding Amount ($)",
        margin={"r": 40, "t": 40, "l": 40, "b": 120},
        height=600
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# 📊 Call the function using the existing dropdown
plot_subagency_obligated_vs_outlayed(dataset, agency_input)


def obligation_ratio_plot(df, agency_name):
    agency_data = df[df["awarding_agency_name"] == agency_name]
    # Define bins and labels
    bins = [-float("inf"), -1, -0.5, 0, 0.5, 1, float("inf")]
    labels = ["< -1", "-1 to -0.5", "-0.5 to 0", "0 to 0.5", "0.5 to 1", "> 1"]

    # Categorize obligation utilization ratio
    agency_data["obligation_utilization_category"] = pd.cut(
        agency_data["obligation_utilization_ratio"], bins=bins, labels=labels
    )

    # Count values for each category
    category_counts = agency_data["obligation_utilization_category"].value_counts().reindex(labels, fill_value=0)
    category_counts_df = category_counts.reset_index()
    category_counts_df.columns = ["Category", "Count"]

    # Create Plotly bar chart
    fig = px.bar(
        category_counts_df,
        x="Category",
        y="Count",
        title="⚠️ Distribution of Obligation Utilization Ratio Categories",
        labels={"Category": "Obligation Utilization Ratio Range", "Count": "Number of Grants"},
        text_auto=True
    )

    # Display in Streamlit
    st.plotly_chart(fig)


obligation_ratio_plot(dataset, agency_input)


def plot_total_vs_anomalous_grants(df, agency_name):
    """
    Generates a pie chart showing the proportion of total grants vs. anomalous grants
    for a selected agency in Streamlit.

    Parameters:
        df (DataFrame): Federal spending dataset.
        agency_name (str): The selected agency name.

    Returns:
        Displays an interactive Plotly pie chart in Streamlit.
    """

    # Filter data for the selected agency
    agency_data = df[df["awarding_agency_name"] == agency_name]

    if agency_data.empty:
        st.warning(f"No records found for '{agency_name}'.")
        return

    total_counts = len(agency_data)

    # Identify anomalies where performance start date is greater than award issue date
    anomalies = agency_data[
        (agency_data["period_of_performance_start_date"] < agency_data["award_base_action_date"]) |
        (agency_data["award_base_action_date"] > agency_data["period_of_performance_current_end_date"])
        ]

    anomalies_filtered_counts = len(anomalies)

    # Data for Pie Chart
    labels = ["Valid Grants", "Anomalous Grants"]
    values = [total_counts - anomalies_filtered_counts, anomalies_filtered_counts]
    colors = ["steelblue", "red"]

    # Create Pie Chart
    fig = px.pie(
        names=labels,
        values=values,
        title=f"🔍 Total vs Anomalous Grants for {agency_name}",
        color=labels,
        color_discrete_map={"Valid Grants": "steelblue", "Anomalous Grants": "red"},
        hole=0.4
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# 📊 Call the function using the existing dropdown
plot_total_vs_anomalous_grants(dataset, agency_input)

#Vlad's viz
# Step 1: Filter for awarding agencies with at least 10 instances
eligible_agencies = dataset["awarding_agency_name"].value_counts()
eligible_agencies = eligible_agencies[eligible_agencies >= 10].index

filtered_df = dataset[dataset["awarding_agency_name"].isin(eligible_agencies)]
def plot_award_distribution(agency_name):
    """
    Generates a Plotly bar chart showing the distribution of awards by month for the selected agency.

    Parameters:
        agency_name (str): The selected awarding agency.

    Returns:
        Displays a bar chart in Streamlit.
    """

    # Filter data for the selected agency
    agency_data = filtered_df[filtered_df["awarding_agency_name"] == agency_name]

    if agency_data.empty:
        st.warning(f"No data found for '{agency_name}'.")
        return

    # Convert award_base_action_date to datetime if not already
    agency_data["award_base_action_date"] = pd.to_datetime(agency_data["award_base_action_date"], errors='coerce')

    # Drop rows with invalid dates
    agency_data = agency_data.dropna(subset=["award_base_action_date"])

    # Extract month from the date
    agency_data["award_month"] = agency_data["award_base_action_date"].dt.month

    # Count occurrences per month
    month_counts = agency_data["award_month"].value_counts().reindex(range(1, 13), fill_value=0).reset_index()
    month_counts.columns = ["Month", "Number of Awards"]

    # Define month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_counts["Month"] = month_counts["Month"].apply(lambda x: month_labels[x - 1])

    # Create Plotly bar chart
    fig = px.bar(
        month_counts,
        x="Month",
        y="Number of Awards",
        title=f"📅 Award Distribution by Month for {agency_name}",
        labels={"Month": "Month", "Number of Awards": "Number of Awards"},
        text_auto=True,
        color_discrete_sequence=["#1f77b4"],  # Match dashboard theme
    )

    # Update layout for consistency
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Awards",
        margin={"r": 40, "t": 40, "l": 40, "b": 60},
        height=500
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# 📊 Call the function using the existing dropdown
# st.subheader("📆 Award Distribution Over Months")
plot_award_distribution(agency_input)