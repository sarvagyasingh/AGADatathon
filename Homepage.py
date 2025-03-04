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
st.title("📊 Federal Spending Transparency Platform")
st.markdown("""
🚀 **Objective:** Improve public access to federal spending data and highlight inconsistencies in fund allocation.

## 🔹 What We’re Building
🖥️ **Interactive Data Dashboard** to analyze, visualize and extract federal spending data from **usaspending.gov**.  

### 📊 **Key Features**
- **State and Agency Insights** – Federal funding analysis across agencies at state level.
- **Historic Trends** – Fund allocation trends and tracking.  
- **Anomaly Tracking** – Identify and visualize inconsistencies or gaps in spending data.  
- **Dynamic Visualizations** – Generate charts, maps, and analytics for deeper insights.  

## 🎯 **Why It Matters?**
- **Enhances public trust** by making federal spending more transparent.  
- **Empowers stakeholders** to track fund utilization and detect anomalies.  
- **Provides actionable insights** for policymakers and watchdogs.  

---
""")



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
            'period_of_performance_current_end_date',
            'prime_award_summary_place_of_performance_cd_original',
            'primary_place_of_performance_state_name',
            'cfda_numbers_and_titles'
        ]
        return pd.read_csv(output, usecols=selected_columns, low_memory=False)

    team_members = [
        {
            "name": "Bhavya More",
            "role": "Product Manager",
            "image": "team_assets/bhavya.jpg",
            "linkedin": "https://www.linkedin.com/in/bhavya-more-582230124/"
        },
        {
            "name": "Megha Kalal",
            "role": "Developer",
            "image": "team_assets/Megha.jpg",
            "linkedin": "https://www.linkedin.com/in/megha-kalal-a6061a166/"
        },
        {
            "name": "Sai Shashank Kudkuli",
            "role": "Analyst",
            "image": "team_assets/Shashank.png",
            "linkedin": "https://www.linkedin.com/in/saishashankk/"
        },
        {
            "name": "Sarvagya Singh",
            "role": "Developer",
            "image": "team_assets/Sarvagya.jpg",
            "linkedin": "https://www.linkedin.com/in/sarvagyasingh1/"
        },
        {
            "name": "Vladimir Martirosyan",
            "role": "Analyst",
            "image": "team_assets/Vlad.jpeg",
            "linkedin": "https://www.linkedin.com/in/vladimir-martirosyan-a9b120226/"
        },
    ]

    # Add Team Section to Homepage
    st.markdown("## 🚀 Meet the Team")

    IMAGE_SIZE = 150

    # Create a row for team members
    cols = st.columns(len(team_members))

    for idx, member in enumerate(team_members):
        with cols[idx]:
            st.image(member["image"], width=150)
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h4 style="margin: 10px 0px 5px;">
                        <a href="{member['linkedin']}" target="_blank" style="text-decoration: none; color: white;">
                            {member['name']}
                        </a>
                    </h4>
                    <p style="font-weight: bold; margin: 0;">{member['role']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            #st.write(member["bio"])

    st.markdown("""
                
                
    """, unsafe_allow_html=True)


    st.info("🔍 Select a page from the sidebar to get started!")

    # Load dataset in the background
    st.session_state["dataset"] = load_data()
    st.session_state["dataset_loaded"] = True