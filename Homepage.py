import streamlit as st
import os
import pandas as pd
import gdown

# Main app entry point
st.set_page_config(
    page_title="Federal Spending Transparency Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    @media screen and (max-width: 768px) {
        .mobile-notice {
            display: block;
            background-color: #ffcc00;
            color: black;
            text-align: center;
            padding: 10px;
            font-weight: bold;
            border-radius: 5px;
            width: 100%;
            position: relative;  /* Ensures it's within content flow */
            top: 10px; /* Pushes it down slightly */
            z-index: 999;
            margin-bottom: 10px;
            margin-top: 60px;  /* Creates space below the Streamlit nav bar */
        }
    }

    @media screen and (min-width: 769px) {
        .mobile-notice {
            display: none;
        }
    }
    </style>

    <div class="mobile-notice">
        📱 For a better experience, please rotate your device to landscape mode!
    </div>
    """,
    unsafe_allow_html=True
)



# Title and introduction
st.title("📊 Federal Spending Transparency Dashboard")
st.markdown("""
🚀 **Objective:** Improve public access to federal spending data and highlight inconsistencies in fund allocation.

## 🔹 What We’re Building
🖥️ **Interactive Streamlit Dashboard** to analyze and visualize federal spending data from **usaspending.gov**.  

### 📊 **Key Features**
- **📌 State-Level Insights** – View federal funding distribution across states.  
- **🏛️ Agency & Sub-Agency Trends** – Track how funds are allocated by different government entities.  
- **⚠️ Discrepancy Detection** – Identify and visualize inconsistencies or gaps in spending data.  
- **📈 Dynamic Visualizations** – Generate charts, maps, and analytics for deeper insights.  

## 🎯 **Why It Matters?**
✅ **Enhances public trust** by making federal spending more transparent.  
✅ **Empowers stakeholders** to track fund utilization and detect anomalies.  
✅ **Provides actionable insights** for policymakers and watchdogs.  

---
""")

st.info("🔍 Select a page from the sidebar to get started!")

# Preload dataset in session state
if "dataset" not in st.session_state:
    st.session_state["dataset_loaded"] = False


    @st.cache_data
    def load_data():
        """Loads the dataset with necessary columns for key metrics."""
        file_id = "1mD5VjHz5zO4Ou3sS7H4BDrIvJ-gQIo8m"
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "grant_combined.csv"

        if not os.path.exists(output):
            gdown.download(url, output, quiet=True)

        selected_columns = [
            "assistance_award_unique_key",
            "award_id_fain",
            "award_id_uri",
            "total_obligated_amount",
            "total_outlayed_amount",
            "total_funding_amount",
            "award_base_action_date",
            "award_latest_action_date",
            "awarding_agency_name",
            "recipient_name",
            "awarding_sub_agency_name",
            'recipient_state_name',
            'recipient_state_code',
            'period_of_performance_start_date',
            'period_of_performance_current_end_date'
        ]
        return pd.read_csv(output, usecols=selected_columns, low_memory=False)


    # Load dataset in the background
    st.session_state["dataset"] = load_data()
    st.session_state["dataset_loaded"] = True