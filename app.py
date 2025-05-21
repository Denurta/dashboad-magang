# --- Import Library ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering # Import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import numpy as np

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

def perform_dbscan(df_scaled, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(df_scaled)
    return clusters, dbscan

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
    plt.xlabel('Jumlah Klaster')
    plt.ylabel('Inertia')
    plt.title('Metode Elbow')
    st.pyplot(plt.gcf())
    plt.clf()
    st.info("\U0001F4CC Titik elbow terbaik adalah pada jumlah klaster di mana penurunan inertia mulai melambat secara signifikan.")

def perform_anova(df, features, cluster_col):
    anova_results = []
    for feature in features:
        groups = [df[df[cluster_col] == k][feature] for k in df[cluster_col].unique()]
        # Filter out empty groups which can happen if a cluster has no data points
        groups = [g for g in groups if not g.empty]
        if len(groups) > 1: # ANOVA requires at least two groups
            f_stat, p_value = f_oneway(*groups)
            anova_results.append({"Variabel": feature, "F-Stat": f_stat, "P-Value": p_value})
        else:
            anova_results.append({"Variabel": feature, "F-Stat": np.nan, "P-Value": np.nan}) # Handle cases with 0 or 1 group
    return pd.DataFrame(anova_results)


def dunn_index(df_scaled, labels):
    distances = squareform(pdist(df_scaled, metric='euclidean'))
    unique_clusters = np.unique(labels)
    intra_cluster_distances = []
    inter_cluster_distances = []

    for cluster in unique_clusters:
        points_in_cluster = df_scaled[labels == cluster]
        if len(points_in_cluster) > 1:
            intra_cluster_distances.append(np.max(pdist(points_in_cluster)))

    if not intra_cluster_distances: # Handle case where all clusters have only one point or are empty
        return np.nan

    for i in range(len(unique_clusters)):
        for j in range(i + 1, len(unique_clusters)):
            cluster_i = df_scaled[labels == unique_clusters[i]]
            cluster_j = df_scaled[labels == unique_clusters[j]]
            if len(cluster_i) > 0 and len(cluster_j) > 0:
                # Calculate minimum distance between points in different clusters
                # This needs to be clarified to calculate minimum distance between points from different clusters
                # A more robust way would be to calculate all pairwise distances between points in cluster_i and cluster_j
                min_dist_inter = np.min(pdist(np.vstack((cluster_i, cluster_j))))
                inter_cluster_distances.append(min_dist_inter)


    if inter_cluster_distances and intra_cluster_distances:
        return np.min(inter_cluster_distances) / np.max(intra_cluster_distances)
    return np.nan

# --- Sidebar & Bahasa ---
st.sidebar.title("\u26f4 Clustering Terminal")
language = st.sidebar.radio("Pilih Bahasa", ["Indonesia", "English"])

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
        "Parameter DBSCAN (eps)": {"Indonesia": "Parameter DBSCAN (eps)", "English": "DBSCAN Parameter (eps)"},
        "Parameter DBSCAN (min_samples)": {"Indonesia": "Parameter DBSCAN (min_samples)", "English": "DBSCAN Parameter (min_samples)"},
        "Parameter Agglomerative (Jumlah Klaster)": {"Indonesia": "Parameter Agglomerative (Jumlah Klaster)", "English": "Agglomerative Parameter (Number of Clusters)"},
        "Parameter Agglomerative (Metode Linkage)": {"Indonesia": "Parameter Agglomerative (Metode Linkage)", "English": "Agglomerative Parameter (Linkage Method)"},
    }
    return translations.get(text, {}).get(language, text)

# --- Sidebar ---
st.sidebar.subheader(translate("Pilih Algoritma Klastering"))
clustering_algorithm = st.sidebar.selectbox("", ["KMeans", "DBSCAN", "Agglomerative Clustering"])

if clustering_algorithm == "KMeans":
    st.sidebar.subheader(translate("Jumlah Klaster"))
    n_clusters = st.sidebar.slider("", 2, 10, 3, key="kmeans_clusters")
elif clustering_algorithm == "DBSCAN": # DBSCAN
    st.sidebar.subheader(translate("Parameter DBSCAN (eps)"))
    eps = st.sidebar.slider("", 0.1, 2.0, 0.5, step=0.1, key="dbscan_eps")
    st.sidebar.subheader(translate("Parameter DBSCAN (min_samples)"))
    min_samples = st.sidebar.slider("", 2, 10, 5, key="dbscan_min_samples")
else: # Agglomerative Clustering
    st.sidebar.subheader(translate("Parameter Agglomerative (Jumlah Klaster)"))
    n_clusters_agg = st.sidebar.slider("", 2, 10, 3, key="agg_clusters")
    st.sidebar.subheader(translate("Parameter Agglomerative (Metode Linkage)"))
    linkage_method = st.sidebar.selectbox("", ["ward", "complete", "average", "single"], key="agg_linkage")


st.sidebar.subheader(translate("Pilih Visualisasi"))
visualization_options = st.sidebar.multiselect("", ["Heatmap", "Boxplot", "Barchart"])

st.sidebar.subheader(translate("Pilih Evaluasi Klaster"))
cluster_evaluation_options = st.sidebar.multiselect("", ["ANOVA", "Silhouette Score", "Dunn Index"])

st.sidebar.subheader(translate("Hapus Baris"))
drop_names = st.sidebar.text_area(translate("Masukkan nama baris yang akan dihapus (pisahkan dengan koma)"), key="drop_names")
drop_button = st.sidebar.button(translate("Hapus Baris"))

# --- Tampilan Utama ---
st.title(translate("Analisis Klaster Terminal"))

# --- Panduan Penggunaan ---
with st.expander("\u2139\uFE0F Panduan Penggunaan Aplikasi" if language == "Indonesia" else "\u2139\uFE0F Application Usage Guide"):
    if language == "Indonesia":
        st.markdown("""
        <ol>
            <li><b>Upload File Excel:</b> Klik tombol <i>"Browse files"</i> untuk mengunggah file data Anda (format <code>.xlsx</code>).</li>
            <li><b>Pilih Variabel:</b> Tentukan variabel mana saja yang ingin digunakan untuk analisis klaster.</li>
            <li><b>Hapus Baris (Opsional):</b> Masukkan nama terminal pada kolom <code>Row Labels</code> yang ingin dihapus, pisahkan dengan koma.</li>
            <li><b>Pilih Algoritma Klastering:</b> Pilih antara KMeans, DBSCAN, atau Agglomerative Clustering. Sesuaikan parameter yang relevan.</li>
            <li><b>Pilih Visualisasi & Evaluasi:</b> Centang visualisasi atau evaluasi klaster yang ingin ditampilkan.</li>
            <li><b>Interpretasi:</b> Hasil akan ditampilkan secara otomatis setelah data dan parameter dimasukkan.</li>
        </ol>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <ol>
            <li><b>Upload Excel File:</b> Click <i>"Browse files"</i> to upload your data file (in <code>.xlsx</code> format).</li>
            <li><b>Select Features:</b> Choose which variables you want to use for cluster analysis.</li>
            <li><b>Remove Rows (Optional):</b> Enter row names from the <code>Row Labels</code> column to be removed, separated by commas.</li>
            <li><b>Select Clustering Algorithm:</b> Choose between KMeans, DBSCAN, or Agglomerative Clustering. Adjust the relevant parameters.</li>
            <li><b>Select Visualizations & Evaluations:</b> Check any cluster visualizations or evaluations you want to see.</li>
            <li><b>Interpretation:</b> The results will be displayed automatically after data and parameters are provided.</li>
        </ol>
        """, unsafe_allow_html=True)

# Upload Data
data_loaded = load_data()
if not data_loaded:
    st.info("\u26A0\uFE0F " + translate("Upload Data untuk Analisis"))

if 'data_uploaded' in st.session_state and st.session_state['data_uploaded']:
    df_cleaned = st.session_state['df_cleaned']

    if 'Row Labels' in df_cleaned.columns:
        if drop_button and drop_names:
            names_to_drop = [name.strip() for name in drop_names.split(',') if name.strip()]
            initial_rows = df_cleaned.shape[0]
            df_cleaned = df_cleaned[~df_cleaned['Row Labels'].isin(names_to_drop)]
            df_cleaned.reset_index(drop=True, inplace=True)
            st.session_state['df_cleaned'] = df_cleaned
            rows_deleted = initial_rows - df_cleaned.shape[0]
            if rows_deleted > 0:
                st.success(f"\u2705 Berhasil menghapus {rows_deleted} baris dengan nama: {names_to_drop}")
            else:
                st.info("Tidak ada baris dengan nama tersebut yang ditemukan.")
    # else: # This else was causing an error message if 'Row Labels' wasn't present and no drop_names were entered
    #     st.error("Kolom 'Row Labels' tidak ditemukan dalam data.")


    if 'df_cleaned' in st.session_state and not st.session_state['df_cleaned'].empty:
        df_cleaned_for_analysis = st.session_state['df_cleaned']
        features = df_cleaned_for_analysis.select_dtypes(include='number').columns.tolist()

        if not features:
            st.error("Tidak ada fitur numerik yang ditemukan dalam data setelah pembersihan. Harap periksa file Excel Anda.")
        else:
            st.subheader(translate("Statistik Deskriptif"))
            st.dataframe(df_cleaned_for_analysis.describe())

            selected_features = st.multiselect("Pilih variabel untuk Klastering", features, default=features)

            if selected_features:
                df_scaled = normalize_data(df_cleaned_for_analysis, selected_features)
                cluster_column_name = ""

                if clustering_algorithm == "KMeans":
                    st.subheader(translate("Metode Elbow"))
                    elbow_method(df_scaled)
                    df_cleaned_for_analysis['KMeans_Cluster'], _ = perform_kmeans(df_scaled, n_clusters)
                    cluster_column_name = 'KMeans_Cluster'
                    st.info(f"KMeans Clustering dengan {n_clusters} klaster.")
                elif clustering_algorithm == "DBSCAN": # DBSCAN
                    df_cleaned_for_analysis['DBSCAN_Cluster'], _ = perform_dbscan(df_scaled, eps, min_samples)
                    cluster_column_name = 'DBSCAN_Cluster'
                    st.info(f"DBSCAN Clustering dengan eps={eps} dan min_samples={min_samples}.")
                    st.warning("Catatan: DBSCAN mungkin menghasilkan klaster '-1' yang menunjukkan titik-titik noise.")
                else: # Agglomerative Clustering
                    df_cleaned_for_analysis['Agglomerative_Cluster'], _ = perform_agglomerative(df_scaled, n_clusters_agg, linkage_method)
                    cluster_column_name = 'Agglomerative_Cluster'
                    st.info(f"Agglomerative Clustering dengan {n_clusters_agg} klaster dan metode linkage '{linkage_method}'.")


                st.subheader(translate("Visualisasi Klaster"))

                if "Heatmap" in visualization_options:
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm')
                    plt.title("Heatmap Korelasi Antar Fitur")
                    st.pyplot(plt.gcf())
                    plt.clf()

                if "Boxplot" in visualization_options and cluster_column_name:
                    num_features = len(selected_features)
                    # Adjust subplot layout for better visualization if many features
                    cols = 2
                    rows = (num_features + cols - 1) // cols
                    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
                    axes = axes.flatten() # Flatten the array of axes for easy iteration

                    for i, feature in enumerate(selected_features):
                        sns.boxplot(x=cluster_column_name, y=feature, data=df_cleaned_for_analysis, ax=axes[i])
                        axes[i].set_title(f"Boxplot: {feature} per Cluster")
                        axes[i].set_xlabel("Cluster")
                        axes[i].set_ylabel(feature)

                    # Remove any unused subplots
                    for j in range(i + 1, len(axes)):
                        fig.delaxes(axes[j])

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.clf()

                if "Barchart" in visualization_options:
                    if 'Row Labels' in df_cleaned_for_analysis.columns:
                        for feature in selected_features:
                            grouped = df_cleaned_for_analysis.groupby('Row Labels')[feature].mean().reset_index()
                            top5 = grouped.nlargest(5, feature)
                            bottom5 = grouped.nsmallest(5, feature)

                            col1, col2 = st.columns(2)

                            with col1:
                                fig_top, ax_top = plt.subplots(figsize=(6, 4)) # Increased figure size
                                sns.barplot(x=feature, y='Row Labels', data=top5, palette='Blues_d', ax=ax_top)
                                ax_top.set_title(f"Top 5 Terminal - {feature}")
                                st.pyplot(fig_top)
                                plt.clf()

                            with col2:
                                fig_bottom, ax_bottom = plt.subplots(figsize=(6, 4)) # Increased figure size
                                sns.barplot(x=feature, y='Row Labels', data=bottom5, palette='Blues_d', ax=ax_bottom)
                                ax_bottom.set_title(f"Bottom 5 Terminal - {feature}")
                                st.pyplot(fig_bottom)
                                plt.clf()
                    else:
                        st.warning("Kolom 'Row Labels' tidak ditemukan pada data untuk visualisasi barchart.")

                st.subheader(translate("Evaluasi Klaster"))
                if cluster_column_name and len(df_cleaned_for_analysis[cluster_column_name].unique()) > 1: # Ensure at least 2 clusters for evaluation metrics
                    if "ANOVA" in cluster_evaluation_options:
                        anova_results = perform_anova(df_cleaned_for_analysis, selected_features, cluster_column_name)
                        st.write(anova_results)
                        interpret = ("\U0001F4CC Interpretasi ANOVA: P-value kurang dari alpha (0.05) menunjukkan terdapat perbedaan signifikan." if language == "Indonesia"
                                     else "\U0001F4CC ANOVA Interpretation: P-value less than alpha (0.05) indicates significant difference.")
                        st.write(interpret if (anova_results["P-Value"] < 0.05).any() else interpret.replace("kurang", "lebih").replace("terdapat", "tidak terdapat"))

                    if "Silhouette Score" in cluster_evaluation_options:
                        # Silhouette score is not suitable for DBSCAN with noise points (-1)
                        # We should filter out noise points for Silhouette calculation
                        if clustering_algorithm == "DBSCAN" and -1 in df_cleaned_for_analysis[cluster_column_name].unique():
                            st.warning("Silhouette Score tidak cocok untuk klastering DBSCAN dengan titik noise (-1). Klaster noise akan dikecualikan dari perhitungan.")
                            non_noise_indices = df_cleaned_for_analysis[df_cleaned_for_analysis[cluster_column_name] != -1].index
                            if len(non_noise_indices) > 1 and len(np.unique(df_cleaned_for_analysis.loc[non_noise_indices, cluster_column_name])) > 1:
                                score = silhouette_score(df_scaled.loc[non_noise_indices], df_cleaned_for_analysis.loc[non_noise_indices, cluster_column_name])
                                st.write(f"*Silhouette Score* (excluding noise): {score:.4f}")
                                if language == "Indonesia":
                                    msg = ("Silhouette Score rendah: klaster kurang baik." if score < 0 else
                                           "Silhouette Score sedang: kualitas klaster sedang." if score <= 0.5 else
                                           "Silhouette Score tinggi: klaster cukup baik.")
                                else:
                                    msg = ("Silhouette Score is low: poor clustering." if score < 0 else
                                           "Silhouette Score is moderate: medium quality clustering." if score <= 0.5 else
                                           "Silhouette Score is high: good clustering.")
                                st.write("\U0001F4CC " + msg)
                            else:
                                st.info("Tidak cukup klaster non-noise untuk menghitung Silhouette Score.")
                        elif len(np.unique(df_cleaned_for_analysis[cluster_column_name])) > 1:
                            score = silhouette_score(df_scaled, df_cleaned_for_analysis[cluster_column_name])
                            st.write(f"*Silhouette Score*: {score:.4f}")
                            if language == "Indonesia":
                                msg = ("Silhouette Score rendah: klaster kurang baik." if score < 0 else
                                       "Silhouette Score sedang: kualitas klaster sedang." if score <= 0.5 else
                                       "Silhouette Score tinggi: klaster cukup baik.")
                            else:
                                msg = ("Silhouette Score is low: poor clustering." if score < 0 else
                                       "Silhouette Score is moderate: medium quality clustering." if score <= 0.5 else
                                       "Silhouette Score is high: good clustering.")
                            st.write("\U0001F4CC " + msg)
                        else:
                            st.info("Tidak cukup klaster (minimal 2) untuk menghitung Silhouette Score.")


                    if "Dunn Index" in cluster_evaluation_options:
                        # Similar to Silhouette, Dunn Index might be affected by noise points in DBSCAN
                        if clustering_algorithm == "DBSCAN" and -1 in df_cleaned_for_analysis[cluster_column_name].unique():
                            st.warning("Dunn Index tidak cocok untuk klastering DBSCAN dengan titik noise (-1). Klaster noise akan dikecualikan dari perhitungan.")
                            non_noise_indices = df_cleaned_for_analysis[df_cleaned_for_analysis[cluster_column_name] != -1].index
                            if len(non_noise_indices) > 1 and len(np.unique(df_cleaned_for_analysis.loc[non_noise_indices, cluster_column_name])) > 1:
                                score = dunn_index(df_scaled.loc[non_noise_indices].to_numpy(), df_cleaned_for_analysis.loc[non_noise_indices, cluster_column_name].to_numpy())
                                st.write(f"*Dunn Index* (excluding noise): {score:.4f}")
                                msg_id = "Dunn Index tinggi: pemisahan antar klaster baik." if score > 1 else "Dunn Index rendah: klaster saling tumpang tindih."
                                msg_en = "Dunn Index is high: good separation between clusters." if score > 1 else "Dunn Index is low: clusters overlap."
                                st.write("\U0001F4CC " + (msg_id if language == "Indonesia" else msg_en))
                            else:
                                st.info("Tidak cukup klaster non-noise untuk menghitung Dunn Index.")
                        elif len(np.unique(df_cleaned_for_analysis[cluster_column_name])) > 1:
                            score = dunn_index(df_scaled.to_numpy(), df_cleaned_for_analysis[cluster_column_name].to_numpy())
                            st.write(f"*Dunn Index*: {score:.4f}")
                            msg_id = "Dunn Index tinggi: pemisahan antar klaster baik." if score > 1 else "Dunn Index rendah: klaster saling tumpang tindih."
                            msg_en = "Dunn Index is high: good separation between clusters." if score > 1 else "Dunn Index is low: clusters overlap."
                            st.write("\U0001F4CC " + (msg_id if language == "Indonesia" else msg_en))
                        else:
                            st.info("Tidak cukup klaster (minimal 2) untuk menghitung Dunn Index.")
                else:
                    st.info("Tidak cukup klaster (minimal 2) atau tidak ada klaster yang terdeteksi untuk evaluasi.")
            else:
                st.warning("Harap pilih setidaknya satu variabel untuk memulai analisis klaster.")
    else:
        st.info("Data telah dihapus atau tidak ada data yang tersisa untuk analisis.")
