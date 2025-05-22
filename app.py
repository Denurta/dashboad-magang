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
    page_title="Pelindo Terminal Analysis",
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
/* Style for top navigation radio buttons */
.stRadio > label {
    font-size: 1.2em; /* Make the label larger */
    font-weight: bold;
    color: #1E3A5F;
}
.stRadio > div[role="radiogroup"] {
    flex-direction: row; /* Arrange radio buttons horizontally */
    gap: 20px; /* Space them out */
    padding-bottom: 20px; /* Add some space below */
    border-bottom: 1px solid #ddd; /* A subtle separator */
    margin-bottom: 20px; /* Space after separator */
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


# --- Translation Function ---
def translate(text):
    translations = {
        "Pilih Bahasa": {"Indonesia": "Pilih Bahasa", "English": "Select Language"},
        "Jumlah Klaster": {"Indonesia": "Jumlah Klaster", "English": "Number of Clusters"},
        "Pilih Visualisasi": {"Indonesia": "Pilih Visualisasi", "English": "Select Visualization"},
        "Pilih Evaluasi Klaster": {"Indonesia": "Pilih Evaluasi Klaster", "English": "Select Cluster Evaluation"},
        "Hapus Baris": {"Indonesia": "Hapus Baris", "English": "Remove Rows"},
        "Masukkan nama baris yang akan dihapus (pisahkan dengan koma)": {"Indonesia": "Masukkan nama nama baris yang akan dihapus (pisahkan dengan koma)", "English": "Enter row names to remove (separate with commas)"},
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
        "Interpretasi Davies-Bouldin Index": {"Indonesia": "Nilai DBI mendekati **0** menunjukkan klaster yang **terpisah dengan baik dan padat**, yang merupakan hasil klasterisasi yang optimal. Semakin tinggi nilai DBI, semakin buruk kualitas klasterisasi (klaster tumpang tindih atau tidak padat).", "English": "Davies-Bouldin Index Interpretation: A DBI value closer to **0** indicates **well-separated and dense clusters**, representing optimal clustering results. A higher DBI value suggests poorer clustering quality (overlapping or non-compact clusters)."},
        "Interpretasi DBI Nilai": {"Indonesia": "Interpretasi: DBI adalah metrik internal, nilai yang **mendekati 0 adalah lebih baik**, menunjukkan klaster yang lebih terpisah dan lebih padat.", "English": "Interpretation: DBI is an internal metric; values **closer to 0 are better**, indicating more separated and denser clusters."},
    }
    return translations.get(text, {}).get(st.session_state.language, text)


# --- Helper Functions for Clustering ---
def load_data():
    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip()
            if 'Row Labels' not in df.columns:
                st.error("Kolom 'Row Labels' tidak ditemukan dalam file Excel. Fitur hapus berdasarkan nama baris tidak akan berfungsi.")
            st.session_state['df_original'] = df
            st.session_state['df_cleaned'] = df.copy()
            st.session_state['data_uploaded'] = True
            return True
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
            return False
    return False

def normalize_data(df, features):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    df_scaled.index = df.index
    return df_scaled

def perform_kmeans(df_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    return clusters, kmeans

def perform_agglomerative(df_scaled, n_clusters_agg, linkage_method):
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters_agg, linkage=linkage_method)
    clusters = agg_clustering.fit_predict(df_scaled)
    return clusters, agg_clustering

def elbow_method(df_scaled):
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, color='steelblue', marker='o', linestyle='-', markersize=8)
    plt.xlabel(translate('Jumlah Klaster'))
    plt.ylabel('Inertia')
    plt.title(translate('Metode Elbow'))
    st.pyplot(plt.gcf())
    plt.clf()
    st.info("\U0001F4CC " + translate("Titik elbow terbaik adalah pada jumlah klaster di mana penurunan inertia mulai melambat secara signifikan. Metode ini paling relevan untuk K-Means."))


def perform_anova(df, features, cluster_col):
    anova_results = []
    for feature in features:
        unique_cluster_labels = [k for k in df[cluster_col].unique()]
        groups = [df[df[cluster_col] == k][feature] for k in unique_cluster_labels]
        groups = [g for g in groups if not g.empty]
        if len(groups) > 1:
            f_stat, p_value = f_oneway(*groups)
            anova_results.append({"Variabel": feature, "F-Stat": f_stat, "P-Value": p_value})
        else:
            anova_results.append({"Variabel": feature, "F-Stat": np.nan, "P-Value": np.nan})
    return pd.DataFrame(anova_results)


# --- Page Functions ---

def home_page():
    st.title("🚢 Welcome to PT Pelindo Terminal Petikemas Surabaya Analysis")

    st.markdown("""
    <div class="home-page-container">
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
    <div class="home-page-container">
        <p>To be a world-class integrated logistics and port ecosystem operator, driving connectivity and economic growth.</p>
    </div>
    """, unsafe_allow_html=True)

    st.header("Our Mission")
    st.markdown("""
    <div class="home-page-container">
        <ul>
            <li>Providing efficient and sustainable port services to support the national logistics ecosystem.</li>
            <li>Developing a robust and innovative port business through synergy and collaboration.</li>
            <li>Creating added value for stakeholders while maintaining environmental sustainability.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.info("Navigate to the 'Clustering Analysis' section to upload your data and perform cluster analysis on terminal metrics.")


# Helper function to handle row deletion logic
def handle_row_deletion_logic():
    # We check if the action flag is set in session state.
    # This flag is set by the button's on_click callback.
    if st.session_state.get('execute_drop_action', False):
        if not st.session_state.data_uploaded:
            st.warning("Silakan unggah data terlebih dahulu untuk menggunakan fitur hapus baris.")
            st.session_state['execute_drop_action'] = False # Reset flag
            return

        if 'Row Labels' not in st.session_state.df_cleaned.columns:
            st.error("Kolom 'Row Labels' tidak ditemukan dalam file Excel. Fitur hapus berdasarkan nama baris tidak akan berfungsi.")
            st.session_state['execute_drop_action'] = False # Reset flag
            return

        current_df = st.session_state.df_cleaned.copy()
        drop_names_str = st.session_state.drop_names_input_val # Get value from session state
        names_to_drop = [name.strip() for name in drop_names_str.split(',') if name.strip()]
        initial_rows = current_df.shape[0]

        if names_to_drop: # Only proceed if names are provided
            df_after_drop = current_df[~current_df['Row Labels'].isin(names_to_drop)].reset_index(drop=True)

            rows_deleted = initial_rows - df_after_drop.shape[0]
            
            if rows_deleted > 0:
                st.session_state['df_cleaned'] = df_after_drop # Update the cleaned DataFrame in session state
                st.success(f"\u2705 Berhasil menghapus {rows_deleted} baris dengan nama: {names_to_drop}")
            else:
                st.info("Tidak ada baris dengan nama tersebut yang ditemukan.")
        else:
            st.warning("Silakan masukkan nama baris yang ingin dihapus.")
            
        # CRUCIAL: Reset the action flag immediately after processing to prevent re-trigger on next rerun
        st.session_state['execute_drop_action'] = False


def clustering_analysis_page_content():
    st.title(translate("Analisis Klaster Terminal"))

    # --- Usage Guide ---
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

    # Upload Data
    data_loaded = load_data()
    if not data_loaded:
        st.info("\u26A0\uFE0F " + translate("Upload Data untuk Analisis"))

    # Process row deletion if button was clicked AND data is loaded
    # This must run before any data is displayed, as it modifies df_cleaned
    # The check for button click is now inside handle_row_deletion_logic itself,
    # which checks the 'execute_drop_action' flag.
    handle_row_deletion_logic()

    # Remaining analysis logic (reads from sidebar state which is set globally)
    # Check df_cleaned for emptiness after potential deletion
    if st.session_state.data_uploaded and not st.session_state['df_cleaned'].empty:
        df_cleaned_for_analysis = st.session_state['df_cleaned']

        features = df_cleaned_for_analysis.select_dtypes(include='number').columns.tolist()

        if not features:
            st.error("Tidak ada fitur numerik yang ditemukan dalam data setelah pembersihan. Harap periksa file Excel Anda.")
        else:
            st.subheader(translate("Statistik Deskriptif"))
            st.dataframe(df_cleaned_for_analysis.describe())

            selected_features = st.multiselect(translate("Pilih Variabel untuk Analisis Klaster"), features, default=features, key="selected_features_all")

            if selected_features:
                df_scaled = normalize_data(df_cleaned_for_analysis, selected_features)

                st.subheader(translate("Metode Elbow"))
                elbow_method(df_scaled)

                cluster_column_name = ""

                # --- CLUSTERING ALGORITHM AND PARAMETERS (READ FROM GLOBAL SIDEBAR STATE) ---
                clustering_algorithm = st.session_state.clustering_algorithm_sidebar # Use global state

                if clustering_algorithm == "KMeans":
                    n_clusters = st.session_state.kmeans_clusters_sidebar # Use global state
                    df_cleaned_for_analysis['KMeans_Cluster'], _ = perform_kmeans(df_scaled, n_clusters)
                    cluster_column_name = 'KMeans_Cluster'
                    st.info(f"KMeans Clustering dengan {n_clusters} klaster.")
                else: # Agglomerative Clustering
                    n_clusters_agg = st.session_state.agg_clusters_sidebar # Use global state
                    linkage_method = st.session_state.agg_linkage_sidebar # Use global state
                    df_cleaned_for_analysis['Agglomerative_Cluster'], _ = perform_agglomerative(df_scaled, n_clusters_agg, linkage_method)
                    cluster_column_name = 'Agglomerative_Cluster'
                    st.info(f"Agglomerative Clustering dengan {n_clusters_agg} klaster dan metode linkage '{linkage_method}'.")

                # --- VISUALIZATION OPTIONS (READ FROM GLOBAL SIDEBAR STATE) ---
                visualization_options = st.session_state.visualization_options_sidebar # Use global state

                st.subheader(translate("Visualisasi Klaster"))

                if "Heatmap" in visualization_options:
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm')
                    plt.title("Heatmap Korelasi Antar Fitur")
                    st.pyplot(plt.gcf())
                    plt.clf()

                if "Boxplot" in visualization_options and cluster_column_name and len(df_cleaned_for_analysis[cluster_column_name].unique()) > 1:
                    num_features = len(selected_features)
                    cols = 2
                    rows = (num_features + cols - 1) // cols
                    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
                    axes = axes.flatten()

                    for i, feature in enumerate(selected_features):
                        sns.boxplot(x=cluster_column_name, y=feature, data=df_cleaned_for_analysis, ax=axes[i])
                        axes[i].set_title(f"Boxplot: {feature} per Cluster")
                        axes[i].set_xlabel("Cluster")
                        axes[i].set_ylabel(feature)

                    for j in range(i + 1, len(axes)):
                        fig.delaxes(axes[j])

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.clf()
                elif "Boxplot" in visualization_options:
                    st.info("Tidak cukup klaster (minimal 2) untuk menampilkan Boxplot." if st.session_state.language == "Indonesia" else "Not enough clusters (minimal 2) to display Boxplot.")


                if "Barchart" in visualization_options:
                    if 'Row Labels' in df_cleaned_for_analysis.columns:
                        for feature in selected_features:
                            grouped = df_cleaned_for_analysis.groupby('Row Labels')[feature].mean().reset_index()
                            top5 = grouped.nlargest(5, feature)
                            bottom5 = grouped.nsmallest(5, feature)

                            col1, col2 = st.columns(2)

                            with col1:
                                fig_top, ax_top = plt.subplots(figsize=(6, 4))
                                sns.barplot(x=feature, y='Row Labels', data=top5, palette='Blues_d', ax=ax_top)
                                ax_top.set_title(f"Top 5 Terminal - {feature}")
                                st.pyplot(fig_top)
                                plt.clf()

                            with col2:
                                fig_bottom, ax_bottom = plt.subplots(figsize=(6, 4))
                                sns.barplot(x=feature, y='Row Labels', data=bottom5, palette='Blues_d', ax=ax_bottom)
                                ax_bottom.set_title(f"Bottom 5 Terminal - {feature}")
                                st.pyplot(fig_bottom)
                                plt.clf()
                    else:
                        st.warning("Kolom 'Row Labels' tidak ditemukan pada data untuk visualisasi barchart." if st.session_state.language == "Indonesia" else "Column 'Row Labels' not found in data for barchart visualization.")

                # --- EVALUATION OPTIONS (READ FROM GLOBAL SIDEBAR STATE) ---
                cluster_evaluation_options = st.session_state.cluster_evaluation_options_sidebar

                st.subheader(translate("Evaluasi Klaster"))
                if cluster_column_name and len(df_cleaned_for_analysis[cluster_column_name].unique()) > 1:
                    if "ANOVA" in cluster_evaluation_options:
                        anova_results = perform_anova(df_cleaned_for_analysis, selected_features, cluster_column_name)
                        st.write(anova_results)
                        interpret = ("\U0001F4CC Interpretasi ANOVA: P-value kurang dari alpha (0.05) menunjukkan terdapat perbedaan signifikan." if st.session_state.language == "Indonesia"
                                      else "\U0001F4CC ANOVA Interpretation: P-value less than alpha (0.05) indicates significant difference.")
                        st.write(interpret if (anova_results["P-Value"] < 0.05).any() else interpret.replace("kurang", "lebih").replace("terdapat", "tidak terdapat"))

                    if "Silhouette Score" in cluster_evaluation_options:
                        if len(np.unique(df_cleaned_for_analysis[cluster_column_name])) > 1:
                            score = silhouette_score(df_scaled, df_cleaned_for_analysis[cluster_column_name])
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
                            st.info("Tidak cukup klaster (minimal 2) untuk menghitung Silhouette Score." if st.session_state.language == "Indonesia" else "Not enough clusters (minimal 2) to calculate Silhouette Score.")

                    if translate("Davies-Bouldin Index") in cluster_evaluation_options:
                        if len(np.unique(df_cleaned_for_analysis[cluster_column_name])) > 1:
                            score = davies_bouldin_score(df_scaled, df_cleaned_for_analysis[cluster_column_name])
                            st.write(f"*{translate('Davies-Bouldin Index')}*: {score:.4f}")
                            st.write("\U0001F4CC " + translate("Interpretasi DBI Nilai")) # Changed to the new, simpler interpretation
                        else:
                            st.info("Tidak cukup klaster (minimal 2) untuk menghitung Davies-Bouldin Index." if st.session_state.language == "Indonesia" else "Not enough clusters (minimal 2) to calculate Davies-Bouldin Index.")
                    else:
                        st.info("Tidak cukup klaster (minimal 2) atau tidak ada klaster yang terdeteksi untuk evaluasi." if st.session_state.language == "Indonesia" else "Not enough clusters (minimal 2) or no clusters detected for evaluation.")
                else:
                    st.warning("Harap pilih setidaknya satu variabel numerik untuk memulai analisis klaster." if st.session_state.language == "Indonesia" else "Please select at least one numeric variable to start cluster analysis.")
    else:
        st.info("Data telah dihapus atau tidak ada data yang tersisa untuk analisis." if st.session_state.language == "Indonesia" else "Data has been removed or no data remaining for analysis.")


# --- Main Application Logic (Page Selection and Sidebar Rendering) ---

# TOP NAVIGATION (Horizontal Radio Buttons)
page_selection = st.radio(
    "Select Page",
    ["Home", "Clustering Analysis"],
    key="top_navigation_radio",
    horizontal=True
)

# Render the sidebar for the current page selection and global controls
st.sidebar.title("Navigation") # Main sidebar title

# Language selector is kept in the sidebar, accessible from both pages
st.sidebar.radio(
    translate("Pilih Bahasa"),
    ["Indonesia", "English"],
    key="language_selector",
    on_change=lambda: st.session_state.__setitem__('language', st.session_state.language_selector)
)

st.sidebar.markdown("---") # Separator

# Conditionally render clustering-specific controls in the sidebar
if page_selection == "Clustering Analysis":
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
    else: # Agglomerative Clustering
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
            st.write(translate("Single"))
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

    # --- "Hapus Baris" Section in Sidebar ---
    st.sidebar.subheader(translate("Hapus Baris"))
    st.sidebar.text_area(
        translate("Masukkan nama baris yang akan dihapus (pisahkan dengan koma)"),
        value=st.session_state.drop_names_input_val,
        key="drop_names_input_val" # This key updates st.session_state.drop_names_input_val
    )
    # The button click sets a flag in session state using on_click
    st.sidebar.button(
        translate("Hapus Baris"),
        key="trigger_drop_button_click", # Use a *new* key for the button to avoid conflict
        on_click=lambda: st.session_state.update(execute_drop_action=True) # Set custom flag for processing
    )


# Display the selected page content
if page_selection == "Home":
    home_page()
elif page_selection == "Clustering Analysis":
    # Call the content function for the clustering page
    # The handle_row_deletion_logic will now correctly check st.session_state.execute_drop_action
    clustering_analysis_page_content()
