# --- Import Library ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN # Added for other models
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import numpy as np
import plotly.express as px # Needed for interactive plots
# from sklearn.decomposition import PCA # Removed as PCA visualization is no longer desired

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
} </style>
""", unsafe_allow_html=True)

# --- Fungsi ---
def load_data():
    """
    Handles file upload and initial data loading into session state.
    """
    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip() # Clean column names
            if 'Row Labels' not in df.columns:
                st.error("Kolom 'Row Labels' tidak ditemukan dalam file Excel. Fitur hapus berdasarkan nama baris tidak akan berfungsi.")
            st.session_state['df_original'] = df # Store original DataFrame
            st.session_state['df_cleaned'] = df.copy() # Store a copy for cleaning operations
            st.session_state['data_uploaded'] = True # Set flag for data upload status
            return True
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
            return False
    return False

def normalize_data(df, features):
    """
    Normalizes selected features using StandardScaler.
    """
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    df_scaled.index = df.index # Preserve original index
    return df_scaled

def perform_clustering(df_scaled, model_name, n_clusters=None, eps=None, min_samples=None):
    """
    Performs clustering using the selected model (KMeans, AgglomerativeClustering, or DBSCAN).

    Args:
        df_scaled (pd.DataFrame): The scaled DataFrame for clustering.
        model_name (str): The name of the clustering model to use.
        n_clusters (int, optional): Number of clusters for KMeans and AgglomerativeClustering.
        eps (float, optional): Maximum distance for DBSCAN.
        min_samples (int, optional): Minimum samples for DBSCAN.

    Returns:
        tuple: A tuple containing the cluster labels and the fitted model.
    """
    model = None
    clusters = None

    if model_name == 'KMeans':
        if n_clusters is None:
            st.error("Number of clusters (n_clusters) is required for KMeans.")
            return None, None
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init is important for KMeans
    elif model_name == 'AgglomerativeClustering':
        if n_clusters is None:
            st.error("Number of clusters (n_clusters) is required for AgglomerativeClustering.")
            return None, None
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif model_name == 'DBSCAN':
        if eps is None or min_samples is None:
            st.error("eps and min_samples are required for DBSCAN.")
            return None, None
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        st.error("Invalid clustering model selected.")
        return None, None

    if model:
        clusters = model.fit_predict(df_scaled)
    return clusters, model

def elbow_method(df_scaled):
    """
    Applies the Elbow Method to help determine the optimal number of clusters for KMeans.
    """
    distortions = []
    K = range(1, 11) # Test for 1 to 10 clusters
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, color='steelblue', marker='o', linestyle='-', markersize=8)
    plt.xlabel('Jumlah Klaster')
    plt.ylabel('Inertia')
    plt.title('Metode Elbow')
    st.pyplot(plt.gcf()) # Display the matplotlib figure in Streamlit
    plt.clf() # Clear the current figure to prevent it from displaying in future runs
    st.info("\U0001F4CC Titik elbow terbaik adalah pada jumlah klaster di mana penurunan inertia mulai melambat secara signifikan.")

def perform_anova(df, features, cluster_column_name):
    """
    Performs one-way ANOVA for each specified feature across the clusters.
    """
    anova_results = []
    for feature in features:
        # Filter out clusters with less than 2 data points or NaNs for robust ANOVA
        valid_clusters = [k for k in df[cluster_column_name].unique() if len(df[df[cluster_column_name] == k]) > 1]
        groups = [df[df[cluster_column_name] == k][feature].dropna() for k in valid_clusters]

        # Only perform ANOVA if there are at least two non-empty groups
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            f_stat, p_value = f_oneway(*groups)
            anova_results.append({"Variabel": feature, "F-Stat": f_stat, "P-Value": p_value})
        else:
            anova_results.append({"Variabel": feature, "F-Stat": np.nan, "P-Value": np.nan}) # Indicate not applicable
    return pd.DataFrame(anova_results)

def dunn_index(df_scaled, labels):
    """
    Calculates the Dunn Index for cluster evaluation.
    A higher Dunn Index indicates better clustering (compact and well-separated clusters).
    Handles DBSCAN noise points (-1 label).
    """
    distances = squareform(pdist(df_scaled, metric='euclidean'))
    unique_clusters = np.unique(labels)
    unique_clusters = unique_clusters[unique_clusters != -1] # Exclude noise points from DBSCAN

    if len(unique_clusters) < 2:
        return np.nan # Dunn index requires at least 2 clusters

    intra_cluster_distances = []
    inter_cluster_distances = []

    # Calculate intra-cluster distances (diameter)
    for cluster_id in unique_clusters:
        points_in_cluster = df_scaled[labels == cluster_id]
        if len(points_in_cluster) > 1:
            intra_cluster_distances.append(np.max(pdist(points_in_cluster)))
        else:
            intra_cluster_distances.append(0) # Diameter of a single point is 0

    # Calculate inter-cluster distances (minimum distance between any two clusters)
    for i in range(len(unique_clusters)):
        for j in range(i + 1, len(unique_clusters)):
            cluster_i_points = df_scaled[labels == unique_clusters[i]]
            cluster_j_points = df_scaled[labels == unique_clusters[j]]

            if len(cluster_i_points) > 0 and len(cluster_j_points) > 0:
                min_dist_between_clusters = np.min(pdist(np.vstack((cluster_i_points, cluster_j_points))))
                inter_cluster_distances.append(min_dist_between_clusters)

    if not inter_cluster_distances or not intra_cluster_distances:
        return np.nan

    max_intra_dist = np.max(intra_cluster_distances)
    if max_intra_dist == 0:
        return np.inf # If all intra-cluster distances are 0, Dunn is infinite
    return np.min(inter_cluster_distances) / max_intra_dist

# --- Sidebar & Bahasa ---
st.sidebar.title("\u26f4 Clustering Terminal")
language = st.sidebar.radio("Pilih Bahasa", ["Indonesia", "English"])

def translate(text):
    """
    Translates text based on the selected language.
    """
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
        "Pilih Model Klastering": {"Indonesia": "Pilih Model Klastering", "English": "Select Clustering Model"},
        "Radius Lingkungan (eps)": {"Indonesia": "Radius Lingkungan (eps)", "English": "Neighborhood Radius (eps)"},
        "Jumlah Sampel Minimum (min_samples)": {"Indonesia": "Jumlah Sampel Minimum (min_samples)", "English": "Minimum Samples (min_samples)"},
        "Jalankan Klastering": {"Indonesia": "Jalankan Klastering", "English": "Run Clustering"},
        "Gagal melakukan klastering. Periksa pengaturan Anda.": {"Indonesia": "Gagal melakukan klastering. Periksa pengaturan Anda.", "English": "Failed to perform clustering. Check your settings."},
        "Silakan pilih setidaknya satu fitur untuk klastering.": {"Indonesia": "Silakan pilih setidaknya satu fitur untuk klastering.", "English": "Please select at least one feature for clustering."},
        "Data yang Diunggah": {"Indonesia": "Data yang Diunggah", "English": "Uploaded Data"},
        "Pilih fitur untuk klastering:": {"Indonesia": "Pilih fitur untuk klastering:", "English": "Select features for clustering:"},
        "Pengaturan Klastering": {"Indonesia": "Pengaturan Klastering", "English": "Clustering Settings"},
        "Hasil Klastering": {"Indonesia": "Hasil Klastering", "English": "Clustering Results"},
        "Tidak dapat membuat visualisasi PCA: {e}. Pastikan data Anda sesuai untuk PCA.": {"Indonesia": "Tidak dapat membuat visualisasi PCA: {e}. Pastikan data Anda sesuai untuk PCA.", "English": "Could not create PCA visualization: {e}. Ensure your data is suitable for PCA."},
        "Skor Siluet": {"Indonesia": "Skor Siluet", "English": "Silhouette Score"},
        "Tidak dapat menghitung skor siluet: {e}": {"Indonesia": "Tidak dapat menghitung skor siluet: {e}", "English": "Could not calculate silhouette score: {e}"},
        "Metode Elbow hanya relevan untuk KMeans.": {"Indonesia": "Metode Elbow hanya relevan untuk KMeans.", "English": "Elbow Method is only relevant for KMeans."},
        "Data berhasil diunggah!": {"Indonesia": "Data berhasil diunggah!", "English": "Data uploaded successfully!"},
        "Pilih fitur untuk evaluasi klaster:": {"Indonesia": "Pilih fitur untuk evaluasi klaster:", "English": "Select features for cluster evaluation:"},
        "Visualisasi (Membutuhkan 'Cluster' Kolom)": {"Indonesia": "Visualisasi (Membutuhkan 'Cluster' Kolom)", "English": "Visualization (Requires 'Cluster' Column)"},
        "Pilih fitur untuk visualisasi Boxplot/Barchart:": {"Indonesia": "Pilih fitur untuk visualisasi Boxplot/Barchart:", "English": "Select features for Boxplot/Barchart visualization:"},
        "Statistik Deskriptif per Klaster": {"Indonesia": "Statistik Deskriptif per Klaster", "English": "Descriptive Statistics per Cluster"},
        "Tidak ada klaster yang ditemukan atau terlalu sedikit klaster untuk visualisasi Heatmap.": {"Indonesia": "Tidak ada klaster yang ditemukan atau terlalu sedikit klaster untuk Heatmap visualization.", "English": "No clusters found or too few clusters for Heatmap visualization."},
        "Pilih fitur untuk Heatmap:": {"Indonesia": "Pilih fitur untuk Heatmap:", "English": "Select features for Heatmap:"},
        "Visualisasi ini membutuhkan kolom 'Cluster'. Silakan jalankan klastering terlebih dahulu.": {"Indonesia": "Visualisasi ini membutuhkan kolom 'Cluster'. Silakan jalankan klastering terlebih dahulu.", "English": "This visualization requires a 'Cluster' column. Please run clustering first."},
        "Tidak ada klaster yang ditemukan atau terlalu sedikit klaster untuk evaluasi.": {"Indonesia": "Tidak ada klaster yang ditemukan atau terlalu sedikit klaster untuk evaluasi.", "English": "No clusters found or too few clusters for evaluation."},
        "Dunn Index": {"Indonesia": "Dunn Index", "English": "Dunn Index"},
        "ANOVA (Analisis Varians)": {"Indonesia": "ANOVA (Analisis Varians)", "English": "ANOVA (Analysis of Variance)"},
        "Tidak dapat menghitung Dunn Index: {e}": {"Indonesia": "Tidak dapat menghitung Dunn Index: {e}", "English": "Could not calculate Dunn Index: {e}"},
        "Tampilkan Metode Elbow": {"Indonesia": "Tampilkan Metode Elbow", "English": "Show Elbow Method"},
        "Kolom 'Row Labels' tidak ditemukan, tidak dapat menghapus berdasarkan nama baris.": {"Indonesia": "Kolom 'Row Labels' tidak ditemukan, tidak dapat menghapus berdasarkan nama baris.", "English": "Column 'Row Labels' not found, cannot remove by row name."},
        "Berhasil menghapus {rows_deleted} baris dengan nama: {names_to_drop}": {"Indonesia": "Berhasil menghapus {rows_deleted} baris dengan nama: {names_to_drop}", "English": "Successfully removed {rows_deleted} rows with names: {names_to_drop}"},
        "Tidak ada baris dengan nama tersebut yang ditemukan.": {"Indonesia": "Tidak ada baris dengan nama tersebut yang ditemukan.", "English": "No rows with such names were found."},
        "P-Value < 0.05 menunjukkan perbedaan signifikan antar klaster untuk variabel tersebut.": {"Indonesia": "P-Value < 0.05 menunjukkan perbedaan signifikan antar klaster untuk variabel tersebut.", "English": "P-Value < 0.05 indicates a significant difference between clusters for that variable."},
        "P-Value > 0.05 menunjukkan tidak terdapat perbedaan signifikan antar klaster untuk variabel tersebut.": {"Indonesia": "P-Value > 0.05 menunjukkan tidak terdapat perbedaan signifikan antar klaster untuk variabel tersebut.", "English": "P-Value > 0.05 indicates no significant difference between clusters for that variable."},
        "Silhouette Score rendah: klaster kurang baik." : {"Indonesia": "Silhouette Score rendah: klaster kurang baik.", "English": "Low Silhouette Score: poor clusters."},
        "Silhouette Score sedang: kualitas klaster sedang." : {"Indonesia": "Silhouette Score sedang: kualitas klaster sedang.", "English": "Moderate Silhouette Score: medium cluster quality."},
        "Silhouette Score tinggi: klaster cukup baik." : {"Indonesia": "Silhouette Score tinggi: klaster cukup baik.", "English": "High Silhouette Score: good clusters."},
        "Dunn Index tinggi: pemisahan antar klaster baik." : {"Indonesia": "Dunn Index tinggi: pemisahan antar klaster baik.", "English": "High Dunn Index: good separation between clusters."},
        "Dunn Index rendah: klaster saling tumpang tindih." : {"Indonesia": "Dunn Index rendah: klaster saling tumpang tindih.", "English": "Low Dunn Index: clusters overlap."},
        "Kolom 'Row Labels' tidak ditemukan pada data.": {"Indonesia": "Kolom 'Row Labels' tidak ditemukan pada data.", "English": "Column 'Row Labels' not found in data."},
        "Pilih variabel untuk Elbow Method": {"Indonesia": "Pilih variabel untuk Elbow Method", "English": "Select variables for Elbow Method"},
        "Pilih fitur untuk menjalankan ANOVA.": {"Indonesia": "Pilih fitur untuk menjalankan ANOVA.", "English": "Select features to run ANOVA."},
        "Tabel Statistik Deskriptif": {"Indonesia": "Tabel Statistik Deskriptif", "English": "Descriptive Statistics Table"}
    }
    return translations.get(text, {}).get(language, text)

# --- Sidebar UI Elements ---
st.sidebar.subheader(translate("Pilih Model Klastering"))
cluster_model_name = st.sidebar.selectbox(
    "",
    ('KMeans', 'AgglomerativeClustering', 'DBSCAN'), # Added AgglomerativeClustering and DBSCAN
    key="cluster_model_select"
)

n_clusters_sidebar = None
eps_sidebar = None
min_samples_sidebar = None

# Dynamically show parameters based on selected model
if cluster_model_name in ['KMeans', 'AgglomerativeClustering']:
    st.sidebar.subheader(translate("Jumlah Klaster"))
    n_clusters_sidebar = st.sidebar.slider("", 2, 10, 3, key="n_clusters_slider")
elif cluster_model_name == 'DBSCAN':
    st.sidebar.subheader(translate("Radius Lingkungan (eps)"))
    eps_sidebar = st.sidebar.slider("", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="eps_slider")
    st.sidebar.subheader(translate("Jumlah Sampel Minimum (min_samples)"))
    min_samples_sidebar = st.sidebar.slider("", min_value=1, max_value=20, value=5, key="min_samples_slider")

st.sidebar.subheader(translate("Pilih Visualisasi"))
# Removed "PCA" from the options
visualization_options = st.sidebar.multiselect("", ["Heatmap", "Boxplot", "Barchart"], key="viz_options")

st.sidebar.subheader(translate("Pilih Evaluasi Klaster"))
cluster_evaluation_options = st.sidebar.multiselect("", ["ANOVA", "Silhouette Score", "Dunn Index"], key="eval_options")

st.sidebar.subheader(translate("Hapus Baris"))
drop_names = st.sidebar.text_area(translate("Masukkan nama baris yang akan dihapus (pisahkan dengan koma)"), key="drop_names_sidebar")
drop_button = st.sidebar.button(translate("Hapus Baris"), key="drop_button_sidebar")


# --- Tampilan Utama ---
st.title(translate("Analisis Klaster Terminal"))

# --- Panduan Penggunaan ---
with st.expander("\u2139\uFE0F Panduan Penggunaan Aplikasi" if language == "Indonesia" else "\u2139\uFE0F Application Usage Guide"):
    if language == "Indonesia":
        st.markdown("""
        <ol>
            <li><b>Upload File Excel:</b> Klik tombol <i>"Browse files"</i> untuk mengunggah file data Anda (format <code>.xlsx</code>).</li>
            <li><b>Pilih Model Klastering:</b> Di sidebar, pilih model klastering yang ingin digunakan (<b>KMeans</b>, <b>AgglomerativeClustering</b>, atau <b>DBSCAN</b>).</li>
            <li><b>Atur Parameter Klastering:</b> Sesuaikan parameter sesuai model yang dipilih (misal: "Jumlah Klaster" untuk KMeans/Agglomerative, atau "Radius Lingkungan (eps)" dan "Jumlah Sampel Minimum (min_samples)" untuk DBSCAN).</li>
            <li><b>Pilih Variabel:</b> Tentukan variabel numerik mana saja yang ingin digunakan untuk analisis klaster.</li>
            <li><b>Hapus Baris (Opsional):</b> Masukkan nama terminal pada kolom <code>Row Labels</code> yang ingin dihapus, pisahkan dengan koma.</li>
            <li><b>Jalankan Klastering:</b> Klik tombol <i>"Jalankan Klastering"</i> di sidebar.</li>
            <li><b>Pilih Visualisasi & Evaluasi:</b> Centang visualisasi atau evaluasi klaster yang ingin ditampilkan.</li>
            <li><b>Interpretasi:</b> Hasil akan ditampilkan secara otomatis setelah klastering dijalankan.</li>
        </ol>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <ol>
            <li><b>Upload Excel File:</b> Click <i>"Browse files"</i> to upload your data file (in <code>.xlsx</code> format).</li>
            <li><b>Select Clustering Model:</b> In the sidebar, choose the clustering model you want to use (<b>KMeans</b>, <b>AgglomerativeClustering</b>, or <b>DBSCAN</b>).</li>
            <li><b>Adjust Clustering Parameters:</b> Adjust parameters specific to the chosen model (e.g., "Number of Clusters" for KMeans/Agglomerative, or "Neighborhood Radius (eps)" and "Minimum Samples (min_samples)" for DBSCAN).</li>
            <li><b>Select Features:</b> Choose which numerical variables you want to use for cluster analysis.</li>
            <li><b>Remove Rows (Optional):</b> Enter row names from the <code>Row Labels</code> column to be removed, separated by commas.</li>
            <li><b>Run Clustering:</b> Click the <i>"Run Clustering"</i> button in the sidebar.</li>
            <li><b>Select Visualizations & Evaluations:</b> Check any cluster visualizations or evaluations you want to see.</li>
            <li><b>Interpretation:</b> The results will be displayed automatically after clustering is performed.</li>
        </ol>
        """, unsafe_allow_html=True)

# Main Application Logic
def app_main():
    # Initialize session state variables if they don't exist
    if 'data_uploaded' not in st.session_state:
        st.session_state['data_uploaded'] = False
    if 'df_original' not in st.session_state:
        st.session_state['df_original'] = pd.DataFrame()
    if 'df_cleaned' not in st.session_state:
        st.session_state['df_cleaned'] = pd.DataFrame()
    if 'clusters_available' not in st.session_state:
        st.session_state['clusters_available'] = False
    if 'selected_features_for_clustering' not in st.session_state:
        st.session_state['selected_features_for_clustering'] = []

    # Section: Upload Data
    data_loaded = load_data()
    if not data_loaded:
        st.info("\u26A0\uFE0F " + translate("Upload Data untuk Analisis"))
        return # Exit the function if no data is loaded yet

    if st.session_state['data_uploaded']:
        df_cleaned = st.session_state['df_cleaned'].copy() # Work with a copy to avoid direct modification issues

        st.subheader(translate("Data yang Diunggah"))
        st.dataframe(df_cleaned.head())

        # Section: Handle row dropping
        if drop_button and drop_names:
            if 'Row Labels' in df_cleaned.columns:
                names_to_drop = [name.strip() for name in drop_names.split(',') if name.strip()]
                initial_rows = df_cleaned.shape[0]
                df_cleaned = df_cleaned[~df_cleaned['Row Labels'].isin(names_to_drop)]
                df_cleaned.reset_index(drop=True, inplace=True)
                st.session_state['df_cleaned'] = df_cleaned.copy() # Update the cleaned DataFrame in session state
                rows_deleted = initial_rows - df_cleaned.shape[0]
                if rows_deleted > 0:
                    st.success(translate(f"\u2705 Berhasil menghapus {rows_deleted} baris dengan nama: {names_to_drop}"))
                    st.dataframe(df_cleaned.head()) # Show updated dataframe after dropping rows
                else:
                    st.info(translate("Tidak ada baris dengan nama tersebut yang ditemukan."))
            else:
                st.error(translate("Kolom 'Row Labels' tidak ditemukan, tidak dapat menghapus berdasarkan nama baris."))

        st.markdown("---")
        st.subheader(translate("Pengaturan Klastering"))

        # Feature selection for clustering
        all_numerical_columns = df_cleaned.select_dtypes(include=np.number).columns.tolist()
        if 'Row Labels' in all_numerical_columns:
            all_numerical_columns.remove('Row Labels') # Exclude 'Row Labels' from features for clustering

        selected_features_for_clustering = st.multiselect(
            translate("Pilih fitur untuk klastering:"),
            all_numerical_columns,
            default=st.session_state['selected_features_for_clustering'] if st.session_state['selected_features_for_clustering'] else all_numerical_columns[:min(5, len(all_numerical_columns))],
            key="feature_select"
        )
        st.session_state['selected_features_for_clustering'] = selected_features_for_clustering


        if selected_features_for_clustering:
            df_selected_for_clustering = df_cleaned[selected_features_for_clustering]
            df_scaled = normalize_data(df_selected_for_clustering, selected_features_for_clustering)

            # Elbow Method for KMeans only
            if cluster_model_name == 'KMeans':
                st.sidebar.markdown("---")
                st.sidebar.subheader(translate("Metode Elbow"))
                if st.sidebar.button(translate("Tampilkan Metode Elbow"), key="elbow_button"):
                    elbow_method(df_scaled)
            else:
                st.sidebar.info(translate("Metode Elbow hanya relevan untuk KMeans."))

            # Run Clustering Button
            if st.sidebar.button(translate("Jalankan Klastering"), key="run_clustering_button"):
                st.write(f"Menjalankan klastering dengan model: **{cluster_model_name}**")
                clusters, model = perform_clustering(df_scaled, cluster_model_name, n_clusters_sidebar, eps_sidebar, min_samples_sidebar)

                if clusters is not None:
                    df_cleaned['Cluster'] = clusters # Assign cluster labels
                    st.session_state['df_cleaned'] = df_cleaned.copy() # Update session state
                    st.session_state['clusters_available'] = True

                    st.subheader(f"{translate('Hasil Klastering')} ({cluster_model_name})")
                    display_columns = ['Row Labels', 'Cluster'] if 'Row Labels' in df_cleaned.columns else ['Cluster'] + selected_features_for_clustering
                    st.dataframe(df_cleaned[display_columns].head())
                    st.write("### Jumlah Anggota Klaster:")
                    st.dataframe(df_cleaned['Cluster'].value_counts().sort_index().rename('Count'))

                else:
                    st.error(translate("Gagal melakukan klastering. Periksa pengaturan Anda."))
                    st.session_state['clusters_available'] = False
            else:
                # If clustering button not pressed, ensure 'Cluster' column is removed for a clean state
                if 'Cluster' in df_cleaned.columns:
                    del df_cleaned['Cluster']
                    st.session_state['df_cleaned'] = df_cleaned.copy()
                st.session_state['clusters_available'] = False

        else:
            st.info(translate("Silakan pilih setidaknya satu fitur untuk klastering."))
            st.session_state['clusters_available'] = False

        # --- Visualizations Section ---
        st.markdown("---")
        st.subheader(translate("Visualisasi Klaster"))

        if st.session_state['clusters_available'] and 'Cluster' in st.session_state['df_cleaned'].columns:
            df_with_clusters = st.session_state['df_cleaned']
            # Data used for plotting should be the original (unscaled) features for better interpretability
            # df_scaled_for_viz = normalize_data(df_with_clusters[st.session_state['selected_features_for_clustering']], st.session_state['selected_features_for_clustering']) # This was used for PCA

            # Conditional Visualizations based on user selection
            if visualization_options:
                viz_features = st.multiselect(translate("Pilih fitur untuk visualisasi Boxplot/Barchart:"),
                                              st.session_state['selected_features_for_clustering'],
                                              key="viz_features_select")

                for viz_type in visualization_options:
                    if viz_type == "Heatmap":
                        st.write("### Heatmap Klaster")
                        df_heatmap = df_with_clusters[df_with_clusters['Cluster'] != -1].copy() # Exclude noise cluster
                        if len(df_heatmap['Cluster'].unique()) > 1 and len(st.session_state['selected_features_for_clustering']) > 0:
                            try:
                                # For heatmap, it's often more useful to show means of original (or scaled) features
                                # Using scaled features to show relative importance across clusters
                                df_scaled_for_heatmap = normalize_data(df_heatmap[st.session_state['selected_features_for_clustering']], st.session_state['selected_features_for_clustering'])
                                cluster_means = df_scaled_for_heatmap.groupby(df_heatmap['Cluster']).mean() # Group by the original (non-filtered) cluster labels
                                fig_heatmap = px.imshow(cluster_means,
                                                        labels=dict(x="Fitur", y="Klaster", color="Nilai Rata-rata"),
                                                        x=cluster_means.columns,
                                                        y=[f"Klaster {c}" for c in cluster_means.index],
                                                        title="Rata-rata Fitur per Klaster (Skala)")
                                st.plotly_chart(fig_heatmap)
                            except Exception as e:
                                st.warning(f"Tidak dapat membuat Heatmap: {e}")
                        else:
                            st.info(translate("Tidak ada klaster yang ditemukan atau terlalu sedikit klaster untuk visualisasi Heatmap."))
                    elif viz_type == "Boxplot" and viz_features:
                        st.write("### Boxplot per Klaster")
                        df_boxplot = df_with_clusters[df_with_clusters['Cluster'] != -1].copy() # Exclude noise cluster
                        if len(df_boxplot['Cluster'].unique()) > 0:
                            for feature in viz_features:
                                fig_boxplot = px.box(df_boxplot, x="Cluster", y=feature, title=f"Boxplot {feature} per Klaster")
                                st.plotly_chart(fig_boxplot)
                        else:
                            st.info("Tidak ada klaster valid untuk visualisasi Boxplot.")
                    elif viz_type == "Barchart" and viz_features:
                        st.write("### Barchart Rata-rata per Klaster")
                        df_barchart = df_with_clusters[df_with_clusters['Cluster'] != -1].copy() # Exclude noise cluster
                        if 'Row Labels' in df_barchart.columns and len(df_barchart['Cluster'].unique()) > 0:
                            for feature in viz_features:
                                grouped = df_barchart.groupby('Row Labels')[feature].mean().reset_index()
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
                            st.warning(translate("Kolom 'Row Labels' tidak ditemukan atau tidak ada klaster valid untuk visualisasi Barchart."))
            else:
                st.info("Pilih jenis visualisasi dari sidebar untuk melihat grafik.")

        else:
            st.info(translate("Visualisasi ini membutuhkan kolom 'Cluster'. Silakan jalankan klastering terlebih dahulu."))

        # --- Descriptive Statistics per Cluster Section ---
        st.markdown("---")
        st.subheader(translate("Statistik Deskriptif per Klaster"))
        if st.session_state['clusters_available'] and 'Cluster' in st.session_state['df_cleaned'].columns:
            df_desc = st.session_state['df_cleaned'].copy()
            df_desc = df_desc[df_desc['Cluster'] != -1] # Exclude noise cluster
            if not df_desc.empty and len(df_desc['Cluster'].unique()) > 0:
                st.write("### Rata-rata Fitur per Klaster")
                st.dataframe(df_desc.groupby('Cluster')[st.session_state['selected_features_for_clustering']].mean())
                st.write("### Standar Deviasi Fitur per Klaster")
                st.dataframe(df_desc.groupby('Cluster')[st.session_state['selected_features_for_clustering']].std())
            else:
                st.info("Tidak ada klaster valid yang ditemukan untuk statistik deskriptif.")
        else:
            st.info(translate("Tidak ada klaster yang ditemukan atau terlalu sedikit klaster untuk evaluasi."))


        # --- Cluster Evaluation Section ---
        st.markdown("---")
        st.subheader(translate("Evaluasi Klaster"))
        if st.session_state['clusters_available'] and 'Cluster' in st.session_state['df_cleaned'].columns:
            df_with_clusters = st.session_state['df_cleaned']
            df_scaled_for_eval = normalize_data(df_with_clusters[st.session_state['selected_features_for_clustering']], st.session_state['selected_features_for_clustering'])
            clusters_labels = df_with_clusters['Cluster'].values

            # Filter out noise points for evaluation metrics
            non_noise_indices = clusters_labels != -1
            clusters_labels_filtered = clusters_labels[non_noise_indices]
            df_scaled_for_eval_filtered = df_scaled_for_eval[non_noise_indices]

            unique_clusters_count = len(np.unique(clusters_labels_filtered))

            if unique_clusters_count < 2 or len(clusters_labels_filtered) < 2:
                st.info(translate("Tidak ada klaster valid yang ditemukan (minimal 2 klaster dan 2 sampel) untuk evaluasi metrik ini."))
            else:
                if "Silhouette Score" in cluster_evaluation_options:
                    try:
                        silhouette_avg = silhouette_score(df_scaled_for_eval_filtered, clusters_labels_filtered)
                        st.write(f"**{translate('Skor Siluet')}:** {silhouette_avg:.4f}")
                        if language == "Indonesia":
                            msg = ("Silhouette Score rendah: klaster kurang baik." if silhouette_avg < 0 else
                                   "Silhouette Score sedang: kualitas klaster sedang." if silhouette_avg <= 0.5 else
                                   "Silhouette Score tinggi: klaster cukup baik.")
                        else:
                            msg = ("Silhouette Score is low: poor clustering." if silhouette_avg < 0 else
                                   "Silhouette Score is moderate: medium quality clustering." if silhouette_avg <= 0.5 else
                                   "Silhouette Score is high: good clusters.")
                        st.write("\U0001F4CC " + msg)
                    except Exception as e:
                        st.warning(translate(f"Tidak dapat menghitung skor siluet: {e}"))

                if "Dunn Index" in cluster_evaluation_options:
                    try:
                        dunn = dunn_index(df_scaled_for_eval_filtered.values, clusters_labels_filtered)
                        if not np.isnan(dunn):
                            st.write(f"**{translate('Dunn Index')}:** {dunn:.4f}")
                            msg_id = "Dunn Index tinggi: pemisahan antar klaster baik." if dunn > 1 else "Dunn Index rendah: klaster saling tumpang tindih."
                            msg_en = "Dunn Index is high: good separation between clusters." if dunn > 1 else "Dunn Index is low: clusters overlap."
                            st.write("\U0001F4CC " + (msg_id if language == "Indonesia" else msg_en))
                        else:
                            st.info(translate("Dunn Index tidak dapat dihitung (mungkin terlalu sedikit klaster valid atau titik data)."))
                    except Exception as e:
                        st.warning(translate(f"Tidak dapat menghitung Dunn Index: {e}"))

                if "ANOVA" in cluster_evaluation_options:
                    eval_features = st.multiselect(translate("Pilih fitur untuk evaluasi klaster:"),
                                                   st.session_state['selected_features_for_clustering'],
                                                   key="anova_features_select")
                    if eval_features:
                        st.write(f"### {translate('ANOVA (Analisis Varians)')}")
                        df_anova = df_with_clusters[df_with_clusters['Cluster'] != -1].copy() # Exclude noise cluster
                        anova_df = perform_anova(df_anova, eval_features, 'Cluster')
                        st.dataframe(anova_df)
                        if not anova_df.empty and (anova_df["P-Value"] < 0.05).any():
                            st.info(translate("P-Value < 0.05 menunjukkan perbedaan signifikan antar klaster untuk variabel tersebut."))
                        else:
                            st.info(translate("P-Value > 0.05 menunjukkan tidak terdapat perbedaan signifikan antar klaster untuk variabel tersebut."))
                    else:
                        st.info(translate("Pilih fitur untuk menjalankan ANOVA."))
        else:
            st.info(translate("Tidak ada klaster yang ditemukan atau terlalu sedikit klaster untuk evaluasi."))


# Run the main app function
if __name__ == '__main__':
    app_main()
