import streamlit as st

st.set_page_config(
    page_title="Pelindo Terminal Analysis",
    page_icon="🚢",
    layout="wide"
)

# Streamlit automatically finds .py files in the 'pages' directory
# and lists them in the sidebar based on their file names.
# No explicit code is needed here to list them if they are in 'pages/'.

# You can add a brief introduction or a redirect if you like,
# but the navigation will appear in the sidebar automatically.
# For example, you could add:
# st.write("### Welcome to the Pelindo Terminal Analysis Application!")
# st.write("Use the sidebar to navigate between the Home page and the Clustering Analysis.")
