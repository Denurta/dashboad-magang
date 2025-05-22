import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np

# --- SET PAGE CONFIG (MUST BE AT THE VERY TOP) ---
st.set_page_config(
    page_title="SPTP Analysis", # Nama aplikasi di tab browser
    page_icon="🚢",
    layout="wide"
)

# --- Styling CSS ---
st.markdown(""" <style>
.stApp {
    background: linear-gradient(to right, rgba(135, 206, 250, 0.4), rgba(70, 130, 180, 0.4));
    color: #1E3A5F;
}
.stSidebar {
    background-color: rgba(50, 90, 140, 0.7);
    color: #2c5b7b;
    border-radius: 10px;
    padding: 10px;
}
.stTextInput, .stSlider, .stButton, .stFileUploader {
    background-color: rgba(255, 255, 255, 0.5);
    color: #1E3A5F;
    border-radius: 10px;
    padding: 10px;
}
h1, h2, h3, h4, h5, h6 {
    color: #1E3A5F;
}
li {
    padding: 5px;
    font-size: 16px;
}
/* Specific styling for the Home page when part of a single file */
.home-page-container {
    background-color: rgba(255, 255, 255, 0.7);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}

/* Container for top navigation buttons to center them */
.centered-buttons-wrapper {
    display: flex;
    justify-content: center; /* Centers the content (which is the st.columns div) */
    width: 100%; /* Ensures the wrapper takes full width */
    margin-bottom: 20px; /* Space below the buttons */
}

/* Style for individual top navigation buttons within the wrapper */
.centered-buttons-wrapper .stButton > button {
    font-size: 1.2em;
    font-weight: bold;
    color: white; /* Text color white for red button */
    background-color: #DC3545; /* Red color (Bootstrap 'danger' red) */
    border: 2px solid #DC3545; /* Red border */
    border-radius: 5px;
    padding: 10px 25px; /* Increase padding for slightly larger buttons */
    box-shadow: 2px 2px 5px rgba(0,0,0,0.2); /* Subtle shadow */
    transition: background-color 0.3s ease, border-color 0.3s ease, transform 0.2s ease; /* Smooth transition */
}
.centered-buttons-wrapper .stButton > button:hover {
    background-color: #C82333; /* Darker red on hover */
    border-color: #C82333;
    transform: translateY(-2px); /* Slight lift effect */
}
/* Adjust spacing for columns if needed, though gap on wrapper or column itself is better */
/* Removed specific Streamlit internal class targeting for robustness */
/* .centered-buttons-wrapper .st-emotion-cache-1pxczgcb {
    gap: 30px;
} */
/* Better way to apply gap between columns if using st.columns within a flex container */
.centered-buttons-wrapper > div > div { /* This targets the columns divs directly */
    gap: 30px; /* Add gap between columns if necessary */
}

</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
# Initialize ALL session state variables used by widgets
# This ensures they exist on the very first run
if 'language' not in st.session_state: st.session_state.language = "Indonesia"
if 'data_uploaded' not in st.session_state: st.session_state.data_uploaded = False
if 'df_original' not in st.session_state: st.session_state.df_original = pd.DataFrame()
if 'df_cleaned' not in st.session_state: st.session_state.df_cleaned = pd.DataFrame()

# Initialize session state for sidebar widgets, including the "Hapus Baris" text_area
if 'clustering_algorithm_sidebar' not in st.session_state: st.session_state.clustering_algorithm_sidebar = "KMeans"
if 'kmeans_clusters_sidebar' not in st.session_state: st.session_state.kmeans_clusters_sidebar = 2
if 'agg_clusters_sidebar' not in st.session_state: st.session_state.agg_clusters_sidebar = 2
if 'agg_linkage_sidebar' not in st.session_state: st.session_state.agg_linkage_sidebar = "ward"
if 'visualization_options_sidebar' not in st.session_state: st.session_state.visualization_options_sidebar = []
if 'cluster_evaluation_options_sidebar' not in st.session_state: st.session_state.cluster_evaluation_options_sidebar = []
if 'drop_names_input_val' not in st.session_state: st.session_state.drop_names_input_val = '' # Initialize text_area value
# New flag for button click action
if 'execute_drop_action' not in st.session_state: st.session_state.execute_drop_action = False
if 'current_page' not in st.session_state: st.session_state.current_page = "Home" # New state for page navigation

# --- Translation Function ---
def translate(text):
    translations = {
        "Pilih Bahasa": {"Indonesia": "Pilih Bahasa", "English": "Select Language"},
        "Jumlah Klaster": {"Indonesia": "Jumlah Klaster", "English": "Number of Clusters"},
        "Pilih Visualisasi": {"Indonesia": "Pilih Visualisasi", "English": "Select Visualization"},
        "Pilih Evaluasi Klaster": {"Indonesia": "Pilih Evaluasi Klaster", "English": "Select Cluster Evaluation"},
        "Hapus Baris": {"Indonesia": "Hapus Baris", "English": "Remove Rows"},
        "Masukkan nama baris yang akan dihapus (pisahkan dengan koma)": {"Indonesia": "Masukkan nama baris yang akan dihapus (pisahkan dengan koma)", "English": "Enter row names to remove (separate with commas)"},
        "Analisis Klaster Terminal": {"Indonesia": "Analisis Klaster Terminal", "English": "Terminal Cluster Analysis"},
        "Metode Elbow": {"Indonesia": "Metode Elbow", "English": "Elbow Method"},
        "Visualisasi Klaster": {"Indonesia": "Visualisasi Klaster", "English": "Cluster Visualization"},
        "Statistik Deskriptif": {"Indonesia": "Statistik Deskriptif", "English": "Descriptive Statistics"},
        "Evaluasi Klaster": {"Indonesia": "Evaluasi Klaster", "English": "Cluster Evaluation"},
        "Upload Data untuk Analisis": {"Indonesia": "Upload Data untuk Analisis", "English": "Upload Data for Analysis"},
        "Pilih Algoritma Klastering": {"Indonesia": "Pilih Algoritma Klastering", "English": "Select Clustering Algorithm"},
        "Parameter Agglomerative (Jumlah Klaster)": {"Indonesia": "Parameter Agglomerative (Jumlah Klaster)", "English": "Agglomerative Parameter (Number of Clusters)"},
        "Parameter Agglomerative (Metode Linkage)": {"Indonesia": "Parameter Agglomerative (Metode Linkage)", "English": "Agglomerative Parameter (Linkage Method)"},
        "Parameter KMeans (Jumlah Klaster)": {"Indonesia": "Parameter KMeans (Jumlah Klaster)", "English": "KMeans Parameter (Number of Clusters)"},
        "Pilih Variabel untuk Analisis Klaster": {"Indonesia": "Pilih Variabel untuk Analisis Klaster", "English": "Select Variables for Cluster Analysis"},
        "Penjelasan Metode Linkage": {"Indonesia": "Penjelasan Metode Linkage", "English": "Explanation of Linkage Methods"},
        "Ward": {"Indonesia": "**Ward:** Menggabungkan klaster yang meminimalkan peningkatan varians internal. Cenderung menghasilkan klaster yang seimbang dan padat. Baik sebagai titik awal." , "English": "**Ward:** Merges clusters that minimize the increase in internal variance. Tends to produce balanced and compact clusters. Good starting point."},
        "Complete": {"Indonesia": "**Complete (Maximum Linkage):** Mengukur jarak maksimum antar dua titik dari klaster berbeda. Baik untuk klaster yang sangat terpisah dan padat, sensitif terhadap outlier.", "English": "**Complete (Maximum Linkage):** Measures the maximum distance between two points from different clusters. Good for very separate and dense clusters, sensitive to outliers."},
        "Average": {"Indonesia": "**Average (Average Linkage):** Mengukur jarak rata-rata antar setiap pasangan titik dari klaster berbeda. Pilihan seimbang, kurang sensitif terhadap outlier.", "English": "**Average (Average Linkage):** Measures the average distance between every pair of points from different clusters. A balanced choice, less sensitive to outliers."},
        "Single": {"Indonesia": "**Single (Minimum Linkage):** Mengukur jarak minimum antar dua titik dari klaster berbeda. Baik untuk klaster berbentuk aneh, tetapi rentan terhadap efek rantai dan outlier.", "English": "**Single (Minimum Linkage):** Measures the minimum distance between two points from different clusters. Good for finding oddly-shaped clusters, but prone to chaining effect and sensitive to outliers."},
        "Davies-Bouldin Index": {"Indonesia": "Davies-Bouldin Index", "English": "Davies-Bouldin Index"},
        "Interpretasi Davies-Bouldin Index": { # Updated interpretation
            "Indonesia": "Nilai DBI yang mendekati 0 adalah lebih baik, menunjukkan klaster yang lebih terpisah dan lebih padat. Indeks ini mengukur rasio antara dispersi intra-klaster (seberapa rapat titik dalam satu klaster) dan jarak antar-klaster (seberapa jauh klaster satu sama lain).",
            "English": "DBI values closer to 0 are better, indicating more separated and denser clusters. This index measures the ratio of within-cluster dispersion (how compact points are within a cluster) and between-cluster separation (how far clusters are from each other)."
        },
        
        # --- TEKS UNTUK FOKUS KE SPTP ---
        "Welcome to SPTP Analysis": {"Indonesia": "Selamat Datang di Analisis SPTP", "English": "Welcome to SPTP Analysis"},
        "About SPTP": {"Indonesia": "Tentang SPTP", "English": "About SPTP"},
        "About SPTP Text 1": {
            "Indonesia": "Sebagai bagian dari integrasi Pelindo, <code> Subholding Pelindo Terminal Petikemas (SPTP)</code> adalah operator terminal terkemuka di Indonesia yang berfokus pada pelayanan peti kemas. Berdirinya SPTP adalah inisiatif strategis untuk mewujudkan konektivitas nasional dan jaringan ekosistem logistik yang lebih kuat, khususnya dalam layanan peti kemas.",
            "English": "As part of Pelindo's integration, <code> Subholding Pelindo Terminal Petikemas (SPTP)</code> is a leading terminal operator in Indonesia focusing on container services. SPTP's establishment is a strategic initiative to realize stronger national connectivity and logistics ecosystem networks, specifically within container services."
        },
        "About SPTP Text 2": {
            "Indonesia": "SPTP memainkan peran krusial dalam rantai logistik nasional dengan mengelola dan mengoperasikan terminal peti kemas di berbagai pelabuhan strategis di seluruh Indonesia. Terminal ini berfungsi sebagai gerbang vital perdagangan, memfasilitasi aliran barang ke dan dari berbagai wilayah secara efisien dan aman.",
            "English": "SPTP plays a crucial role in the national logistics chain by managing and operating container terminals across various strategic ports in Indonesia. These terminals serve as vital trade gateways, facilitating the efficient and safe flow of goods to and from various regions."
        },
        "About SPTP Text 3": {
            "Indonesia": "Dengan kendali strategis yang lebih baik dan kemampuan finansial yang kuat, operasional bisnis SPTP menjadi lebih terkoordinasi, terstandar, dan efisien, memberikan keuntungan bagi masyarakat dan pengguna jasa. Komitmen kami adalah menyediakan layanan terminal peti kemas yang unggul dan handal, mendukung pertumbuhan ekonomi, dan meningkatkan daya saing Indonesia dalam perdagangan global.",
            "English": "With improved strategic control and strong financial capabilities, SPTP's business operations are more coordinated, standardized, and efficient, benefiting both the public and service users. Our commitment is to provide excellent and reliable container terminal services, supporting economic growth, and enhancing Indonesia's competitiveness in global trade."
        },
        
        "Our Vision": {"Indonesia": "Visi", "English": "Vision"},
        "Vision Text": {"Indonesia": "Operator terminal terkemuka yang berkelas dunia", "English": "A leading world-class terminal operator"},
        "Our Mission": {"Indonesia": "Misi", "English": "Mission"},
        "Mission Item 1": {"Indonesia": "Mendukung ekosistem petikemas yang terintegrasi melalui keunggulan operasional", "English": "Supporting an integrated container ecosystem through operational excellence"},
        "Mission Item 2": {"Indonesia": "Optimalisasi jaringan", "English": "Network optimization"},
        "Mission Item 3": {"Indonesia": "Kemitraan strategis untuk pertumbuhan ekonomi nasional", "English": "Strategic partnerships for national economic growth"},
        "Home": {"Indonesia": "Beranda", "English": "Home"},
        "Clustering Analysis": {"Indonesia": "Analisis Klastering", "English": "Clustering Analysis"},
        
        # Text for the "Terminal Performance Analysis" section
        "Terminal Performance Analysis Title": {"Indonesia": "📊 Analisis Kinerja Terminal", "English": "📊 Terminal Performance Analysis"},
        "Analysis Objective Text": {
            "Indonesia": "Dalam upaya mendukung pengambilan keputusan berbasis data, analisis ini bertujuan untuk mengelompokkan terminal peti kemas berdasarkan kinerja operasional menggunakan algoritma <code>K-Means</code> dan <code>Agglomerative Clustering</code>, serta mengevaluasi perbedaan antar kelompok melalui Analisis <code>ANOVA</code>.",
            "English": "To support data-driven decision-making, this analysis aims to cluster container terminals based on operational performance using the <code>K-Means</code> and <code>Agglomerative Clustering</code> algorithm, and evaluate differences between groups through <code>ANOVA</code> Analysis."
        },
        "Performance Variables Title": {"Indonesia": "⚙️ Variabel Kinerja yang Dianalisis", "English": "⚙️ Performance Variables Analyzed"},
        "ET/BT Variable": {
            "Indonesia": "🔹 <code>ET/BT (Efisiensi Waktu Operasional)</code> : Rasio antara waktu efektif dan waktu sandar kapal. Semakin tinggi, semakin efisien aktivitas bongkar muat.",
            "English": "🔹 <code>ET/BT (Operational Time Efficiency)</code> : Ratio between effective time and ship's berth time. Higher values indicate more efficient loading/unloading activities."
        },
        "BSH/BT Variable": {
            "Indonesia": "🔹 <code>BSH/BT (Produktivitas Waktu Sandar)</code> : Mengukur berapa banyak petikemas yang dibongkar per jam selama kapal berada di dermaga.",
            "English": "🔹 <code>BSH/BT (Berth Time Productivity)</code> : Measures how many containers are unloaded per hour per hour while the ship is at the berth."
        },
        "BCH/ET Variable": {
            "Indonesia": "🔹 <code>BCH/ET (Produktivitas Waktu Efektif)</code> : Menunjukkan produktivitas per jam dalam waktu kerja yang benar-benar digunakan untuk operasi.",
            "English": "🔹 <code>BCH/ET (Effective Time Productivity)</code> : Indicates productivity per hour in actual operational working time."
        },
        "Standardization Note": {
            "Indonesia": "Sebelum analisis dilakukan, semua variabel distandarisasi agar memiliki skala yang setara, sehingga proses pengelompokan dapat dilakukan secara objektif.",
            "English": "Before analysis, all variables are standardized to have an equivalent scale, enabling objective clustering."
        },
        "Methodology Title": {"Indonesia": "🧠 Metodologi Pengelompokan", "English": "🧠 Clustering Methodology"},
        "Methodology Item Clustering": { # Combined K-Means and Agglomerative
            "Indonesia": "💡 <code>K-Means</code> dan <code>Agglomerative Clustering</code> digunakan untuk mengelompokkan terminal dengan karakteristik operasional yang serupa.",
            "English": "💡 <code>K-Means</code> and <code>Agglomerative Clustering</code> are used to group terminals with similar operational characteristics."
        },
        "Methodology Item Elbow": {
            "Indonesia": "📉 <code>Metode Elbow</code> membantu menentukan jumlah klaster yang optimal.",
            "English": "📉 <code>Elbow Method</code> helps determine the optimal number of clusters."
        },
        "Methodology Item Evaluation Metrics": { # Combined Silhouette and DBI
            "Indonesia": "✅ <code>Silhouette Score</code> dan <code>Davies-Bouldin Index (DBI)</code> dihitung untuk mengevaluasi seberapa baik terminal dikelompokkan.",
            "English": "✅ <code>Silhouette Score</code> and <code>Davies-Bouldin Index (DBI)</code> are calculated to evaluate how well terminals are grouped."
        },
        "Methodology Item ANOVA": {
            "Indonesia": "🧪 <code>Uji ANOVA</code> dilakukan untuk melihat apakah terdapat perbedaan signifikan antar klaster pada masing-masing variabel kinerja.",
            "English": "🧪 <code>ANOVA test</code> is performed to see if there are significant differences between clusters for each performance variable."
        },
        "Analysis Objective Section Title": {"Indonesia": "🎯 Tujuan Analisis", "English": "🎯 Analysis Objective"},
        "Analysis Objective Item 1": {
            "Indonesia": "Mengidentifikasi pola performa terminal berdasarkan data operasional.",
            "English": "Identify terminal performance patterns based on operational data."
        },
        "Analysis Objective Item 2": {
            "Indonesia": "Menyediakan insight strategis untuk mendukung peningkatan layanan.",
            "English": "Provide strategic insights to support service improvement."
        },
        "Analysis Objective Item 3": {
            "Indonesia": "Membantu pengelola pelabuhan dalam mengambil keputusan berbasis data, misalnya peningkatan infrastruktur atau alokasi sumber daya.",
            "English": "Assist port managers in making evidence-based decisions, such as improving infrastructure, or allocating resources."
        },
    }
    return translations.get(text, {}).get(st.session_state.language, text)


# --- Helper Functions for Clustering ---

# Use st.cache_data for functions that return dataframes/objects that don't change often
# This helps prevent reloading/reprocessing data on every rerun if inputs are the same
@st.cache_data
def load_and_process_data(uploaded_file):
    """Loads and performs initial processing of the Excel file."""
    if uploaded_file is None:
        return pd.DataFrame(), False, "No file uploaded."

    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()
        
        # Check for 'Row Labels' early
        if 'Row Labels' not in df.columns:
            st.error("Kolom 'Row Labels' tidak ditemukan dalam file Excel. Fitur hapus berdasarkan nama baris tidak akan berfungsi.")
            return pd.DataFrame(), False, "Missing 'Row Labels' column."

        # It's okay to store original and a copy for cleaning in session state,
        # as load_and_process_data is cached and only runs when file changes.
        st.session_state['df_original'] = df
        st.session_state['df_cleaned'] = df.copy() # Make a copy here, will modify df_cleaned
        return df, True, ""
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
        return pd.DataFrame(), False, f"Error reading file: {e}"

@st.cache_data
def normalize_data(df, features):
    """Scales the selected features."""
    if df.empty or not features:
        return pd.DataFrame()
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    df_scaled.index = df.index # Maintain original index for joining later
    return df_scaled

@st.cache_data
def perform_kmeans(df_scaled, n_clusters):
    """Performs KMeans clustering."""
    if df_scaled.empty:
        return np.array([]), None
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    return clusters, kmeans

@st.cache_data
def perform_agglomerative(df_scaled, n_clusters_agg, linkage_method):
    """Performs Agglomerative Clustering."""
    if df_scaled.empty:
        return np.array([]), None
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters_agg, linkage=linkage_method)
    clusters = agg_clustering.fit_predict(df_scaled)
    return clusters, agg_clustering

def elbow_method(df_scaled):
    """Displays the Elbow Method plot."""
    if df_scaled.empty:
        st.info(translate("Harap unggah data dan pilih variabel untuk melihat Metode Elbow."))
        return

    distortions = []
    K = range(1, min(11, len(df_scaled) + 1)) # Max K is num of samples
    if len(K) < 2: # Need at least 2 points for elbow
        st.info(translate("Tidak cukup data untuk menampilkan Metode Elbow (minimal 2 sampel)."))
        return

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        distortions.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K, distortions, color='steelblue', marker='o', linestyle='-', markersize=8)
    ax.set_xlabel(translate('Jumlah Klaster'))
    ax.set_ylabel('Inertia')
    ax.set_title(translate('Metode Elbow'))
    st.pyplot(fig)
    plt.close(fig) # Explicitly close the figure to free memory
    st.info("\U0001F4CC " + translate("Titik elbow terbaik adalah pada jumlah klaster di mana penurunan inertia mulai melambat secara signifikan. Metode ini paling relevan untuk K-Means."))


@st.cache_data
def perform_anova(df, features, cluster_col):
    """Performs ANOVA test for each feature across clusters."""
    if df.empty or not features or cluster_col not in df.columns:
        return pd.DataFrame()
    
    anova_results = []
    for feature in features:
        unique_cluster_labels = [k for k in df[cluster_col].unique()]
        groups = [df[df[cluster_col] == k][feature].dropna() for k in unique_cluster_labels]
        # Filter out empty groups and groups with single unique value (ANOVA requires variance)
        groups = [g for g in groups if len(g) > 1 and g.nunique() > 1] 

        if len(groups) > 1:
            f_stat, p_value = f_oneway(*groups)
            anova_results.append({"Variabel": feature, "F-Stat": f_stat, "P-Value": p_value})
        else:
            anova_results.append({"Variabel": feature, "F-Stat": np.nan, "P-Value": np.nan}) # No valid groups for ANOVA
    return pd.DataFrame(anova_results)


# --- Page Functions ---

def home_page():
    # Judul utama aplikasi, sekarang fokus pada SPTP
    st.title("🚢 " + translate("Welcome to SPTP Analysis"))

    st.markdown(f"""
    <div class="home-page-container">
        <h3>{translate("About SPTP")}</h3>
        <p>
            {translate("About SPTP Text 1")}
        </p>
        <p>
            {translate("About SPTP Text 2")}
        </p>
        <p>
            {translate("About SPTP Text 3")}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.header(translate("Our Vision"))
    st.markdown(f"""
    <div class="home-page-container">
        <p>{translate("Vision Text")}</p>
    </div>
    """, unsafe_allow_html=True)

    st.header(translate("Our Mission"))
    st.markdown(f"""
    <div class="home-page-container">
        <ul>
            <li>{translate("Mission Item 1")}</li>
            <li>{translate("Mission Item 2")}</li>
            <li>{translate("Mission Item 3")}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # --- NEW SECTION: Terminal Performance Analysis ---
    st.header(translate("Terminal Performance Analysis Title"))
    st.markdown(f"""
    <div class="home-page-container">
        <p>{translate("Analysis Objective Text")}</p>
        <h4>{translate("Performance Variables Title")}</h4>
        <ul>
            <li>{translate("ET/BT Variable")}</li>
            <li>{translate("BSH/BT Variable")}</li>
            <li>{translate("BCH/ET Variable")}</li>
        </ul>
        <p>{translate("Standardization Note")}</p>
        <h4>{translate("Methodology Title")}</h4>
        <ul>
            <li>{translate("Methodology Item Clustering")}</li>
            <li>{translate("Methodology Item Elbow")}</li>
            <li>{translate("Methodology Item Evaluation Metrics")}</li>
            <li>{translate("Methodology Item ANOVA")}</li>
        </ul>
        <h4>{translate("Analysis Objective Section Title")}</h4>
        <ul>
            <li>{translate("Analysis Objective Item 1")}</li>
            <li>{translate("Analysis Objective Item 2")}</li>
            <li>{translate("Analysis Objective Item 3")}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    # --- END NEW SECTION ---

    st.info(translate("Navigate to the 'Clustering Analysis' section to upload your data and perform cluster analysis on terminal metrics."))


# Helper function to handle row deletion logic
def handle_row_deletion_logic():
    # Only execute if the button was clicked and data is available
    if st.session_state.get('execute_drop_action', False):
        if not st.session_state.data_uploaded or st.session_state['df_original'].empty:
            st.warning("Silakan unggah data terlebih dahulu untuk menggunakan fitur hapus baris.")
            st.session_state['execute_drop_action'] = False # Reset flag
            return

        # Ensure 'Row Labels' column exists in the original data to perform deletion
        if 'Row Labels' not in st.session_state.df_original.columns:
            st.error("Kolom 'Row Labels' tidak ditemukan dalam file Excel asli. Fitur hapus berdasarkan nama baris tidak akan berfungsi.")
            st.session_state['execute_drop_action'] = False # Reset flag
            return

        drop_names_str = st.session_state.drop_names_input_val.strip()
        names_to_drop = [name.strip() for name in drop_names_str.split(',') if name.strip()]
        
        if not names_to_drop:
            st.warning("Silakan masukkan nama baris yang ingin dihapus.")
            st.session_state['execute_drop_action'] = False
            return

        # Perform deletion on df_original.copy() and update df_cleaned
        # This ensures df_original remains untouched and df_cleaned reflects changes
        initial_rows = st.session_state.df_cleaned.shape[0]
        
        # Using .loc for direct modification and avoiding excessive copies
        rows_before_drop = st.session_state.df_cleaned.shape[0]
        st.session_state.df_cleaned = st.session_state.df_cleaned[~st.session_state.df_cleaned['Row Labels'].isin(names_to_drop)].reset_index(drop=True)
        rows_deleted = rows_before_drop - st.session_state.df_cleaned.shape[0]
            
        if rows_deleted > 0:
            st.success(f"\u2705 Berhasil menghapus {rows_deleted} baris dengan nama: {', '.join(names_to_drop)}")
            # Clear selected features if any removed rows impact them
            # This helps avoid errors if a removed row was critical for a selected feature's calculation
            # For simplicity, we just trigger a rerun if deletion happens, other widgets will react
            st.rerun() 
        else:
            st.info(f"Tidak ada baris dengan nama '{', '.join(names_to_drop)}' yang ditemukan untuk dihapus.")
            
        # CRUCIAL: Reset the action flag immediately after processing to prevent re-trigger on next rerun
        st.session_state['execute_drop_action'] = False


def clustering_analysis_page_content():
    st.title(translate("Analisis Klaster Terminal"))

    with st.expander("\u2139\uFE0F Panduan Penggunaan Aplikasi" if st.session_state.language == "Indonesia" else "\u2139\uFE0F Application Usage Guide"):
        if st.session_state.language == "Indonesia":
            st.markdown("""
            <ol>
                <li><b>Upload File Excel:</b> Klik tombol <i>"Browse files"</i> untuk mengunggah file data Anda (format <code>.xlsx</code>).</li>
                <li><b>Pilih Variabel:</b> Tentukan variabel numerik mana saja yang ingin digunakan untuk analisis klaster (Metode Elbow dan klastering).</li>
                <li><b>Hapus Baris (Opsional):</b> Masukkan nama terminal pada kolom <code>Row Labels</code> yang ingin dihapus, pisahkan dengan koma.</li>
                <li><b>Pilih Algoritma Klastering:</b> Pilih antara <code>KMeans</code> atau <code>Agglomerative Clustering</code>. Sesuaikan parameter yang relevan.</li>
                <li><b>Pilih Visualisasi & Evaluasi:</b> Centang visualisasi atau evaluasi klaster yang ingin ditampilkan.</li>
                <li><b>Interpretasi:</b> Hasil akan ditampilkan secara otomatis setelah data dan parameter dimasukkan.</li>
            </ol>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <ol>
                <li><b>Upload Excel File:</b> Click <i>"Browse files"</i> to upload your data file (in <code>.xlsx</code> format).</li>
                <li><b>Select Variables:</b> Choose which numerical variables you want to use for cluster analysis (Elbow Method and clustering).</li>
                <li><b>Remove Rows (Optional):</b> Enter row names from the <code>Row Labels</code> column to be removed, separated by commas.</li>
                <li><b>Select Clustering Algorithm:</b> Choose between <code>KMeans</code> or <code>Agglomerative Clustering</code>. Adjust the relevant parameters.</li>
                <li><b>Select Visualizations & Evaluations:</b> Check any cluster visualizations or evaluations you want to see.</li>
                <li><b>Interpretation:</b> The results will be displayed automatically after data and parameters are provided.</li>
            </ol>
            """, unsafe_allow_html=True)

    # --- Data Loading and Processing ---
    # Use st.session_state.uploaded_file to prevent re-upload on every rerun
    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"], key="file_uploader_main")
    
    # Only call load_and_process_data if a new file is uploaded or if df_original is empty
    if uploaded_file and (uploaded_file != st.session_state.get('last_uploaded_file_id') or st.session_state['df_original'].empty):
        with st.spinner("Memproses data..."):
            df_loaded, data_status, message = load_and_process_data(uploaded_file)
            if data_status:
                st.session_state['data_uploaded'] = True
                st.session_state['df_original'] = df_loaded
                st.session_state['df_cleaned'] = df_loaded.copy() # Reset df_cleaned if new file
                st.session_state['last_uploaded_file_id'] = uploaded_file # Store file ID to detect changes
                st.success("Data berhasil diunggah dan dimuat!")
            else:
                st.session_state['data_uploaded'] = False
                st.error(message)
    elif not st.session_state.data_uploaded:
        st.info("\u26A0\uFE0F " + translate("Upload Data untuk Analisis"))

    # Handle row deletion logic immediately after loading/uploading
    handle_row_deletion_logic() # This modifies st.session_state.df_cleaned

    # Proceed with analysis only if data is loaded and not empty after cleaning
    if st.session_state.data_uploaded and not st.session_state['df_cleaned'].empty:
        df_current_analysis = st.session_state['df_cleaned']

        # Get numeric features from the current cleaned DataFrame
        features = df_current_analysis.select_dtypes(include='number').columns.tolist()

        if not features:
            st.error("Tidak ada fitur numerik yang ditemukan dalam data setelah pembersihan. Harap periksa file Excel Anda.")
            return # Exit if no features

        st.subheader(translate("Statistik Deskriptif"))
        st.dataframe(df_current_analysis.describe())

        selected_features = st.multiselect(
            translate("Pilih Variabel untuk Analisis Klaster"), 
            features, 
            default=features if features else [], # Default to all if available, else empty list
            key="selected_features_all"
        )

        if not selected_features:
            st.warning("Harap pilih setidaknya satu variabel numerik untuk memulai analisis klaster." if st.session_state.language == "Indonesia" else "Please select at least one numeric variable to start cluster analysis.")
            return # Exit if not enough features selected

        df_scaled = normalize_data(df_current_analysis, selected_features)

        st.subheader(translate("Metode Elbow"))
        elbow_method(df_scaled) # This function handles its own plotting and clearing

        cluster_column_name = ""
        clustering_algorithm = st.session_state.clustering_algorithm_sidebar

        if clustering_algorithm == "KMeans":
            n_clusters = st.session_state.kmeans_clusters_sidebar
            # Ensure n_clusters is valid for the number of samples
            if n_clusters >= len(df_scaled):
                st.warning(f"Jumlah klaster KMeans ({n_clusters}) harus kurang dari jumlah sampel ({len(df_scaled)}). Menggunakan {len(df_scaled)-1} klaster.")
                n_clusters = max(2, len(df_scaled) - 1) # Fallback to a valid number
            
            if n_clusters < 2:
                st.info("Tidak cukup sampel untuk melakukan klastering KMeans (minimal 2 klaster diperlukan).")
                return # Exit if not enough samples
            
            clusters, _ = perform_kmeans(df_scaled, n_clusters)
            df_current_analysis['KMeans_Cluster'] = clusters
            cluster_column_name = 'KMeans_Cluster'
            st.info(f"KMeans Clustering dengan {n_clusters} klaster.")
        else: # Agglomerative Clustering
            n_clusters_agg = st.session_state.agg_clusters_sidebar
            # Ensure n_clusters_agg is valid for the number of samples
            if n_clusters_agg >= len(df_scaled):
                st.warning(f"Jumlah klaster Agglomerative ({n_clusters_agg}) harus kurang dari jumlah sampel ({len(df_scaled)}). Menggunakan {len(df_scaled)-1} klaster.")
                n_clusters_agg = max(2, len(df_scaled) - 1) # Fallback to a valid number

            if n_clusters_agg < 2:
                st.info("Tidak cukup sampel untuk melakukan klastering Agglomerative (minimal 2 klaster diperlukan).")
                return # Exit if not enough samples

            linkage_method = st.session_state.agg_linkage_sidebar
            clusters, _ = perform_agglomerative(df_scaled, n_clusters_agg, linkage_method)
            df_current_analysis['Agglomerative_Cluster'] = clusters
            cluster_column_name = 'Agglomerative_Cluster'
            st.info(f"Agglomerative Clustering dengan {n_clusters_agg} klaster dan metode linkage '{linkage_method}'.")

        # --- VISUALIZATION OPTIONS ---
        visualization_options = st.session_state.visualization_options_sidebar
        st.subheader(translate("Visualisasi Klaster"))

        if "Heatmap" in visualization_options:
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 6))
            sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm', ax=ax_heatmap)
            ax_heatmap.set_title("Heatmap Korelasi Antar Fitur")
            st.pyplot(fig_heatmap)
            plt.close(fig_heatmap) # Explicitly close

        # Only proceed with Boxplot if there are clusters generated and more than one unique cluster
        if "Boxplot" in visualization_options and cluster_column_name and len(df_current_analysis[cluster_column_name].unique()) > 1:
            num_features = len(selected_features)
            cols = 2
            rows = (num_features + cols - 1) // cols
            fig_box, axes_box = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            axes_box = axes_box.flatten()

            for i, feature in enumerate(selected_features):
                sns.boxplot(x=cluster_column_name, y=feature, data=df_current_analysis, ax=axes_box[i])
                axes_box[i].set_title(f"Boxplot: {feature} per Cluster")
                axes_box[i].set_xlabel("Cluster")
                axes_box[i].set_ylabel(feature)

            for j in range(i + 1, len(axes_box)): # Hide unused subplots
                fig_box.delaxes(axes_box[j])

            plt.tight_layout()
            st.pyplot(fig_box)
            plt.close(fig_box) # Explicitly close
        elif "Boxplot" in visualization_options:
            st.info("Tidak cukup klaster (minimal 2) untuk menampilkan Boxplot." if st.session_state.language == "Indonesia" else "Not enough clusters (minimal 2) to display Boxplot.")


        if "Barchart" in visualization_options:
            if 'Row Labels' in df_current_analysis.columns:
                for feature in selected_features:
                    grouped = df_current_analysis.groupby('Row Labels')[feature].mean().reset_index()
                    
                    if not grouped.empty:
                        top5 = grouped.nlargest(5, feature)
                        bottom5 = grouped.nsmallest(5, feature)

                        col1, col2 = st.columns(2)

                        with col1:
                            fig_top, ax_top = plt.subplots(figsize=(6, 4))
                            sns.barplot(x=feature, y='Row Labels', data=top5, palette='Blues_d', ax=ax_top)
                            ax_top.set_title(f"Top 5 Terminal - {feature}")
                            st.pyplot(fig_top)
                            plt.close(fig_top) # Explicitly close

                        with col2:
                            fig_bottom, ax_bottom = plt.subplots(figsize=(6, 4))
                            sns.barplot(x=feature, y='Row Labels', data=bottom5, palette='Blues_d', ax=ax_bottom)
                            ax_bottom.set_title(f"Bottom 5 Terminal - {feature}")
                            st.pyplot(fig_bottom)
                            plt.close(fig_bottom) # Explicitly close
                    else:
                        st.info(f"Tidak ada data untuk membuat Barchart untuk {feature}.")
            else:
                st.warning("Kolom 'Row Labels' tidak ditemukan pada data untuk visualisasi barchart." if st.session_state.language == "Indonesia" else "Column 'Row Labels' not found in data for barchart visualization.")

        # --- EVALUATION OPTIONS ---
        cluster_evaluation_options = st.session_state.cluster_evaluation_options_sidebar
        st.subheader(translate("Evaluasi Klaster"))

        # Check if clustering was performed and has more than one cluster before evaluation
        if cluster_column_name and len(df_current_analysis[cluster_column_name].unique()) > 1:
            if "ANOVA" in cluster_evaluation_options:
                anova_results = perform_anova(df_current_analysis, selected_features, cluster_column_name)
                if not anova_results.empty:
                    st.write(anova_results)
                    interpret = ("\U0001F4CC P-value kurang dari alpha (0.05) menunjukkan terdapat perbedaan signifikan." if st.session_state.language == "Indonesia"
                                  else "\U0001F4CC P-value less than alpha (0.05) indicates significant difference.")
                    st.write(interpret if (anova_results["P-Value"] < 0.05).any() else interpret.replace("kurang", "lebih").replace("terdapat", "tidak terdapat"))
                else:
                    st.info("Tidak ada hasil ANOVA untuk ditampilkan (mungkin tidak cukup variasi data atau klaster).")

            if "Silhouette Score" in cluster_evaluation_options:
                # Ensure enough samples and clusters for Silhouette Score
                if len(df_scaled) > 1 and len(np.unique(df_current_analysis[cluster_column_name])) > 1:
                    score = silhouette_score(df_scaled, df_current_analysis[cluster_column_name])
                    st.write(f"*Silhouette Score*: {score:.4f}")
                    if st.session_state.language == "Indonesia":
                        if score >= 0.71:
                            msg = "Struktur klaster yang dihasilkan sangat kuat. Objek sangat cocok dengan klaster-nya sendiri dan tidak cocok dengan klaster tetangga."
                        elif score >= 0.51:
                            msg = "Struktur klaster yang dihasilkan baik. Objek cocok dengan klaster-nya dan terpisah dengan baik dari klaster lain."
                        elif score >= 0.26:
                            msg = "Struktur klaster yang dihasilkan lemah. Mungkin dapat diterima, tetapi perlu dipertimbangkan bahwa objek mungkin berada di antara klaster."
                        else:
                            msg = "Klaster tidak terstruktur dengan baik. Objek mungkin lebih cocok ditempatkan pada klaster lain daripada klaster saat ini."
                    else: # English
                        if score >= 0.71:
                            msg = "The resulting cluster structure is very strong. Objects fit well within their own cluster and are poorly matched to neighboring clusters."
                        elif score >= 0.51:
                            msg = "The resulting cluster structure is good. Objects fit well within their cluster and are well separated from other clusters."
                        elif score >= 0.26:
                            msg = "The resulting cluster structure is weak. It might be acceptable, but consider that objects might be between clusters."
                        else:
                            msg = "Clusters are not well-structured. Objects might be better placed in another cluster than their current one."
                    st.write("\U0001F4CC " + msg)
                else:
                    st.info("Tidak cukup klaster (minimal 2) atau sampel untuk menghitung Silhouette Score." if st.session_state.language == "Indonesia" else "Not enough clusters (minimal 2) or samples to calculate Silhouette Score.")

            if translate("Davies-Bouldin Index") in cluster_evaluation_options:
                # Ensure enough samples and clusters for Davies-Bouldin Index
                if len(df_scaled) > 1 and len(np.unique(df_current_analysis[cluster_column_name])) > 1:
                    score = davies_bouldin_score(df_scaled, df_current_analysis[cluster_column_name])
                    st.write(f"*{translate('Davies-Bouldin Index')}*: {score:.4f}")
                    st.write("\U0001F4CC " + translate("Interpretasi Davies-Bouldin Index"))
                else:
                    st.info("Tidak cukup klaster (minimal 2) atau sampel untuk menghitung Davies-Bouldin Index." if st.session_state.language == "Indonesia" else "Not enough clusters (minimal 2) or samples to calculate Davies-Bouldin Index.")
        else:
            st.info("Tidak cukup klaster (minimal 2) atau tidak ada klaster yang terdeteksi untuk evaluasi." if st.session_state.language == "Indonesia" else "Not enough clusters (minimal 2) or no clusters detected for evaluation.")
    else: # If df_cleaned is empty after deletion
        st.info("Data telah dihapus atau tidak ada data yang tersisa untuk analisis." if st.session_state.language == "Indonesia" else "Data has been removed or no data remaining for analysis.")


# --- Main Application Logic (Page Selection and Sidebar Rendering) ---

# Pembungkus untuk menengahkan seluruh grup tombol
st.markdown('<div class="centered-buttons-wrapper">', unsafe_allow_html=True)
# Menggunakan st.columns di dalam wrapper untuk menempatkan tombol bersebelahan
col_home, col_clustering = st.columns(2)

with col_home:
    if st.button(translate("Home"), key="btn_home_main"):
        st.session_state.current_page = "Home"
with col_clustering:
    if st.button(translate("Clustering Analysis"), key="btn_clustering_analysis_main"):
        st.session_state.current_page = "Clustering Analysis"
st.markdown('</div>', unsafe_allow_html=True) # Tutup container wrapper

st.markdown("---") # Separator di bawah tombol

# Render sidebar
st.sidebar.title("Navigation")

st.sidebar.radio(
    translate("Pilih Bahasa"),
    ["Indonesia", "English"],
    key="language_selector",
    on_change=lambda: st.session_state.__setitem__('language', st.session_state.language_selector)
)

st.sidebar.markdown("---")

if st.session_state.current_page == "Clustering Analysis":
    st.sidebar.subheader(translate("Pilih Algoritma Klastering"))
    st.sidebar.selectbox(
        "Algoritma", ["KMeans", "Agglomerative Clustering"],
        key="clustering_algorithm_sidebar"
    )

    if st.session_state.clustering_algorithm_sidebar == "KMeans":
        st.sidebar.slider(
            translate("Parameter KMeans (Jumlah Klaster)"), 2, 10,
            value=st.session_state.kmeans_clusters_sidebar,
            key="kmeans_clusters_sidebar"
        )
    else:
        st.sidebar.slider(
            translate("Parameter Agglomerative (Jumlah Klaster)"), 2, 10,
            value=st.session_state.agg_clusters_sidebar,
            key="agg_clusters_sidebar"
        )
        st.sidebar.selectbox(
            translate("Parameter Agglomerative (Metode Linkage)"), ["ward", "complete", "average", "single"],
            key="agg_linkage_sidebar"
        )
        with st.sidebar.expander(translate("Penjelasan Metode Linkage")):
            st.write(translate("Ward"))
            st.write(translate("Complete"))
            st.write(translate("Average"))
            st.write(translate("Single")) # Corrected to use translate for consistency
            if st.session_state.language == "Indonesia":
                st.info("Penting juga untuk diingat bahwa tidak ada satu metrik validasi klaster yang sempurna. Seringkali, kombinasi beberapa metrik dan pemahaman domain data Anda akan memberikan penilaian terbaik terhadap kualitas hasil klasterisasi.")
            else:
                st.info("It is also important to remember that no single cluster validation metric is perfect. Often, a combination of several metrics and understanding your data's domain will provide the best assessment of clustering quality.")

    st.sidebar.subheader(translate("Pilih Visualisasi"))
    st.sidebar.multiselect(
        "Visualisasi", ["Heatmap", "Boxplot", "Barchart"],
        key="visualization_options_sidebar"
    )

    st.sidebar.subheader(translate("Pilih Evaluasi Klaster"))
    st.sidebar.multiselect(
        "Evaluasi", ["ANOVA", "Silhouette Score", translate("Davies-Bouldin Index")],
        key="cluster_evaluation_options_sidebar"
    )

    st.sidebar.subheader(translate("Hapus Baris"))
    # The text_area value is stored in session state, so it persists across reruns
    st.sidebar.text_area(
        translate("Masukkan nama baris yang akan dihapus (pisahkan dengan koma)"),
        value=st.session_state.drop_names_input_val,
        key="drop_names_input_val"
    )
    # The button click sets a flag in session state using on_click
    st.sidebar.button(
        translate("Hapus Baris"),
        key="trigger_drop_button_click",
        on_click=lambda: st.session_state.update(execute_drop_action=True)
    )

# Display the selected page content
if st.session_state.current_page == "Home":
    home_page()
elif st.session_state.current_page == "Clustering Analysis":
    clustering_analysis_page_content()
