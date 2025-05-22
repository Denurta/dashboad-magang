import streamlit as st

# --- Styling CSS (Page Specific - can be made global if preferred) ---
st.markdown(""" <style>
.stApp {
    background: linear-gradient(to right, rgba(135, 206, 250, 0.4), rgba(70, 130, 180, 0.4));
    color: #1E3A5F;
}
h1, h2, h3, h4, h5, h6 {
    color: #1E3A5F;
}
.stExpander {
    background-color: rgba(255, 255, 255, 0.5);
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# --- Page Content ---
st.title("🚢 Welcome to PT Pelindo Terminal Petikemas Surabaya Analysis")

st.markdown("""
<div style="background-color: rgba(255, 255, 255, 0.7); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h3>About PT Pelindo Terminal Petikemas Surabaya</h3>
    <p>
        PT Pelindo Terminal Petikemas (SPTP), a subsidiary of PT Pelabuhan Indonesia (Pelindo), is one of the leading container terminal operators in Indonesia. SPTP plays a crucial role in the national logistics chain by managing and operating container terminals across various strategic ports in Indonesia.
    </p>
    <p>
        Located in Surabaya, East Java, the **Petikemas Surabaya Terminal** serves as a vital gateway for trade, facilitating the flow of goods to and from the eastern part of Indonesia. It is equipped with modern facilities and technology to handle various types of container cargo efficiently and safely.
    </p>
    <p>
        Our commitment is to provide excellent and reliable container terminal services, supporting economic growth, and enhancing Indonesia's competitiveness in global trade.
    </p>
</div>
""", unsafe_allow_html=True)

st.header("Our Vision")
st.markdown("""
<div style="background-color: rgba(255, 255, 255, 0.7); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
    <p>To be a world-class integrated logistics and port ecosystem operator, driving connectivity and economic growth.</p>
</div>
""", unsafe_allow_html=True)

st.header("Our Mission")
st.markdown("""
<div style="background-color: rgba(255, 255, 255, 0.7); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
    <ul>
        <li>Providing efficient and sustainable port services to support the national logistics ecosystem.</li>
        <li>Developing a robust and innovative port business through synergy and collaboration.</li>
        <li>Creating added value for stakeholders while maintaining environmental sustainability.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.info("Navigate to the 'Clustering Analysis' page in the sidebar to upload your data and perform cluster analysis on terminal metrics.")