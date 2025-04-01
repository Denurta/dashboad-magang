import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import numpy as np

# CSS untuk mengubah background dengan tema pelabuhan/laut dengan tampilan lebih transparan
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, rgba(135, 206, 250, 0.4), rgba(70, 130, 180, 0.4));
        color: #57aae2;
    }
    .stSidebar {
        background-color: rgba(50, 90, 140, 0.7); 
        color: #2939af;
        border-radius: 10px;
        padding: 10px;
    }
    .stTextInput, .stSlider, .stButton, .stFileUploader {
        background-color: rgba(255, 255, 255, 0.5);
        color: #57aae2;
        border-radius: 10px;
        padding: 10px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #57aae2;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Fungsi untuk memuat data
def load_data():
    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()
        return df
    return None

# Fungsi untuk normalisasi
def normalize_data(df, features):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    df_scaled.index = df.index
    return df_scaled

# Fungsi untuk K-Means Clustering
def perform_kmeans(df_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    return clusters, kmeans

# Fungsi untuk metode Elbow
def elbow_method(df_scaled):
    distortions = []
    K = range(2, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bo-', markersize=8)
    plt.xlabel('Jumlah Klaster')
    plt.ylabel('Inertia')
    plt.title('Metode Elbow')
    st.pyplot(plt)

# Fungsi untuk ANOVA
def perform_anova(df, features):
    anova_results = []
    for feature in features:
        groups = [df[df['KMeans_Cluster'] == k][feature] for k in df['KMeans_Cluster'].unique()]
        f_stat, p_value = f_oneway(*groups)
        anova_results.append({"Variabel": feature, "F-Stat": f_stat, "P-Value": p_value})
    return pd.DataFrame(anova_results)

# Fungsi untuk menghitung Dunn Index
def dunn_index(df_scaled, labels):
    distances = squareform(pdist(df_scaled, metric='euclidean'))
    unique_clusters = np.unique(labels)

    intra_cluster_distances = []
    inter_cluster_distances = []

    for cluster in unique_clusters:
        points_in_cluster = df_scaled[labels == cluster]
        if len(points_in_cluster) > 1:
            intra_cluster_distances.append(np.max(pdist(points_in_cluster)))

    for i in range(len(unique_clusters)):
        for j in range(i + 1, len(unique_clusters)):
            cluster_i = df_scaled[labels == unique_clusters[i]]
            cluster_j = df_scaled[labels == unique_clusters[j]]
            inter_cluster_distances.append(np.min(pdist(np.vstack((cluster_i, cluster_j)))))

    return np.min(inter_cluster_distances) / np.max(intra_cluster_distances)

# Sidebar dengan judul aplikasi
st.sidebar.title("⛴ Clustering Terminal")

# Pilihan bahasa
language = st.sidebar.radio("Pilih Bahasa", ["Indonesia", "English"])

def translate(text):
    translations = {
        "Jumlah Klaster": {"Indonesia": "Jumlah Klaster", "English": "Number of Clusters"},
        "Pilih Visualisasi": {"Indonesia": "Pilih Visualisasi", "English": "Select Visualization"},
        "Pilih Statistik Deskriptif": {"Indonesia": "Pilih Statistik Deskriptif", "English": "Select Descriptive Statistics"},
        "Pilih Evaluasi Klaster": {"Indonesia": "Pilih Evaluasi Klaster", "English": "Select Cluster Evaluation"},
        "Hapus Baris": {"Indonesia": "Hapus Baris", "English": "Remove Rows"},
        "Masukkan indeks baris yang akan dihapus (pisahkan dengan koma)": {"Indonesia": "Masukkan indeks baris yang akan dihapus (pisahkan dengan koma)", "English": "Enter row indices to remove (separate with commas)"},
        "Analisis Klaster Terminal": {"Indonesia": "Analisis Klaster Terminal", "English": "Terminal Cluster Analysis"},
        "Metode Elbow": {"Indonesia": "Metode Elbow", "English": "Elbow Method"},
        "Visualisasi Klaster": {"Indonesia": "Visualisasi Klaster", "English": "Cluster Visualization"},
        "Statistik Deskriptif": {"Indonesia": "Statistik Deskriptif", "English": "Descriptive Statistics"},
        "Evaluasi Klaster": {"Indonesia": "Evaluasi Klaster", "English": "Cluster Evaluation"},
    }
    return translations.get(text, {}).get(language, text)

n_clusters = st.sidebar.slider(translate("Jumlah Klaster"), min_value=2, max_value=10, value=3)
visualization_options = st.sidebar.multiselect(translate("Pilih Visualisasi"), ["Scatter Plot", "Heatmap", "Boxplot"])
cluster_evaluation_options = st.sidebar.multiselect(translate("Pilih Evaluasi Klaster"), ["ANOVA", "Silhouette Score", "Dunn Index"])

# Opsi untuk menghapus baris tertentu
st.sidebar.subheader(translate("Hapus Baris"))
drop_rows = st.sidebar.text_area(translate("Masukkan indeks baris yang akan dihapus (pisahkan dengan koma)"))

# Tampilan Utama
title_text = translate("Analisis Klaster Terminal")
st.title(title_text)
df = load_data()

if df is not None:
    # Menghapus baris yang dipilih
    if drop_rows:
        drop_indices = [int(i) for i in drop_rows.split(',') if i.isdigit()]
        df = df.drop(index=drop_indices, errors='ignore')
    
    features = df.select_dtypes(include=['number']).columns.tolist()
    
    # Menampilkan Statistik Deskriptif
    st.subheader(translate("Statistik Deskriptif"))
    st.write(df.describe())
    
    # Pilih variabel untuk Elbow Method
    selected_features = st.multiselect("Pilih variabel untuk Elbow Method", features, default=features)
    
    if selected_features:
        df_scaled = normalize_data(df, selected_features)
        
        # Menampilkan metode Elbow sebelum klastering dilakukan
        st.subheader(translate("Metode Elbow"))
        elbow_method(df_scaled)
        
        df['KMeans_Cluster'], kmeans_model = perform_kmeans(df_scaled, n_clusters)
    
    # Visualisasi Klaster
    st.subheader(translate("Visualisasi Klaster"))
    if "Scatter Plot" in visualization_options:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df_scaled.iloc[:, 0], y=df_scaled.iloc[:, 1], hue=df['KMeans_Cluster'], palette='viridis')
        st.pyplot(plt)
    if "Heatmap" in visualization_options:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)
    if "Boxplot" in visualization_options:
        for feature in selected_features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df['KMeans_Cluster'], y=df[feature])
            plt.title(f"Boxplot {feature} per Klaster")
            st.pyplot(plt)
    
 # Evaluasi Klaster
    st.subheader(translate("Evaluasi Klaster"))
    if "ANOVA" in cluster_evaluation_options:
        anova_results = perform_anova(df, selected_features)
        st.write(anova_results)
        if (anova_results["P-Value"] < 0.05).any():
            st.write("📌 " + ("Interpretasi Anova: P-value kurang dari alpha menunjukkan terdapat perbedaan yang signifikan pada variabel tersebut di antara klaster." if language == "Indonesia" else "ANOVA Interpretation: A P-value less than alpha indicates a significant difference in the variable among clusters."))
        else:
            st.write("📌 " + ("Interpretasi Anova: P-value lebih dari alpha menunjukkan tidak terdapat perbedaan yang signifikan pada variabel tersebut di antara klaster." if language == "Indonesia" else "ANOVA Interpretation: A P-value greater than alpha indicates no significant difference in the variable among clusters."))
    if "Silhouette Score" in cluster_evaluation_options:
        silhouette_avg = silhouette_score(df_scaled, df['KMeans_Cluster'])
        st.write(f"Silhouette Score: {silhouette_avg:.4f}")
        if silhouette_avg < 0:
            st.write("📌 " + ("Interpretasi Silhouette Score: Nilai Silhouette Score yang lebih rendah menunjukkan bahwa klaster yang terbentuk kurang baik." if language == "Indonesia" else "Silhouette Score Interpretation: A lower Silhouette Score indicates that the clusters are not well-formed."))
        elif silhouette_avg > 0.5:
            st.write("📌 " + ("Interpretasi Silhouette Score: Nilai Silhouette Score yang lebih tinggi menunjukkan bahwa klaster yang terbentuk cukup baik." if language == "Indonesia" else "Silhouette Score Interpretation: A higher Silhouette Score indicates well-formed clusters."))
        else:
            st.write("📌 " + ("Interpretasi Silhouette Score: Nilai Silhouette Score yang berada di antara 0 dan 0.5 menunjukkan bahwa klaster yang terbentuk memiliki kualitas yang sedang." if language == "Indonesia" else "Silhouette Score Interpretation: A Silhouette Score between 0 and 0.5 indicates moderate cluster quality."))
    if "Dunn Index" in cluster_evaluation_options:
        dunn_idx = dunn_index(df_scaled.to_numpy(), df['KMeans_Cluster'].to_numpy())
        st.write(f"Dunn Index: {dunn_idx:.4f}")
        if dunn_idx > 1:
            st.write("📌 " + ("Interpretasi Dunn Index: Dunn Index yang lebih tinggi menunjukkan pemisahan yang lebih baik antar klaster." if language == "Indonesia" else "Dunn Index Interpretation: A higher Dunn Index indicates better separation between clusters."))
        else:
            st.write("📌 " + ("Interpretasi Dunn Index: Dunn Index yang rendah menunjukkan bahwa klaster yang terbentuk mungkin saling tumpang tindih." if language == "Indonesia" else "Dunn Index Interpretation: A lower Dunn Index indicates possible overlapping clusters."))

else:
    st.warning("⚠ Silakan upload file Excel terlebih dahulu.")
