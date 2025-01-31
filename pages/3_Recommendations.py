import streamlit as st

st.title("Recommendations Page")
st.markdown("""
### Data-Driven Recommendations
Provide recommendations based on the data or user input.

For example:
- Suggest optimal allocation based on spending patterns.
- Highlight areas of concern based on key metrics.

""")

# Example: Add a simple input for user interaction
user_input = st.text_input("Enter a metric or key insight to explore recommendations:")
if user_input:
    st.write(f"Based on your input: **{user_input}**, here are our recommendations:")
    st.markdown("- Suggestion 1")
    st.markdown("- Suggestion 2")
    st.markdown("- Suggestion 3")