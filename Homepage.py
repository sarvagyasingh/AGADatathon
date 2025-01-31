import streamlit as st

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
    </style>
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