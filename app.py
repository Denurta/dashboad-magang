# --- Import Library ---
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

# --- Styling CSS ---
st.markdown("""
    <style>
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
    </style>
""", unsafe_allow_html=True)

# --- Fungsi ---
def load_data():
    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()
        return df
    return None

def normalize_data(df, features):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    df_scaled.index = df.index
    return df_scaled

def perform_kmeans(df_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    return clusters, kmeans

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

def perform_anova(df, features):
    anova_results = []
    for feature in features:
        groups = [df[df['KMeans_Cluster'] == k][feature] for k in df['KMeans_Cluster'].unique()]
        f_stat, p_value = f_oneway(*groups)
        anova_results.append({"Variabel": feature, "F-Stat": f_stat, "P-Value": p_value})
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

    for i in range(len(unique_clusters)):
        for j in range(i + 1, len(unique_clusters)):
            cluster_i = df_scaled[labels == unique_clusters[i]]
            cluster_j = df_scaled[labels == unique_clusters[j]]
            inter_cluster_distances.append(np.min(pdist(np.vstack((cluster_i, cluster_j)))))

    return np.min(inter_cluster_distances) / np.max(intra_cluster_distances)

# --- Sidebar & Bahasa ---
st.sidebar.title("\u26f4 Clustering Terminal")
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

n_clusters = st.sidebar.slider(translate("Jumlah Klaster"), 2, 10, 3)
visualization_options = st.sidebar.multiselect(translate("Pilih Visualisasi"), ["Scatter Plot", "Heatmap", "Boxplot", "Barchart"])
cluster_evaluation_options = st.sidebar.multiselect(translate("Pilih Evaluasi Klaster"), ["ANOVA", "Silhouette Score", "Dunn Index"])

# Hapus baris
st.sidebar.subheader(translate("Hapus Baris"))
drop_rows = st.sidebar.text_area(translate("Masukkan indeks baris yang akan dihapus (pisahkan dengan koma)"))

# --- Tampilan Utama ---
st.title(translate("Analisis Klaster Terminal"))
df = load_data()

if df is not None:
    if drop_rows:
        drop_indices = [int(i) for i in drop_rows.split(',') if i.isdigit()]
        df = df.drop(index=drop_indices, errors='ignore')

    features = df.select_dtypes(include='number').columns.tolist()
    st.subheader(translate("Statistik Deskriptif"))
    st.write(df.describe())

    selected_features = st.multiselect("Pilih variabel untuk Elbow Method", features, default=features)

    if selected_features:
        df_scaled = normalize_data(df, selected_features)
        st.subheader(translate("Metode Elbow"))
        elbow_method(df_scaled)

        df['KMeans_Cluster'], kmeans_model = perform_kmeans(df_scaled, n_clusters)

        # Visualisasi Klaster
        st.subheader(translate("Visualisasi Klaster"))
        if "Scatter Plot" in visualization_options:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df_scaled.iloc[:, 0], y=df_scaled.iloc[:, 1], hue=df['KMeans_Cluster'], palette='viridis')
            st.pyplot(plt.gcf())
            plt.clf()

        if "Heatmap" in visualization_options:
            plt.figure(figsize=(10, 6))
            sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt.gcf())
            plt.clf()

        if "Boxplot" in visualization_options:
            num_features = len(selected_features)
            fig, axes = plt.subplots(1, num_features, figsize=(5 * num_features, 5))
            if num_features == 1:
                axes = [axes]
            for i, feature in enumerate(selected_features):
                sns.boxplot(x='KMeans_Cluster', y=feature, data=df, ax=axes[i])
                axes[i].set_title(f"Boxplot of {feature} per Cluster")
            st.pyplot(fig)
            plt.clf()

        if "Barchart" in visualization_options:
            st.subheader("\U0001F50D Top 5 & Bottom 5 per Variabel")

        for feature in selected_features:
            st.markdown(f"### Variabel: {feature}")

            # Tangani jika data kurang dari 5
            top5 = df[[feature]].nlargest(min(5, len(df)), feature)
            bottom5 = df[[feature]].nsmallest(min(5, len(df)), feature)

            # Label index agar lebih informatif
            top5['Label'] = top5.index.astype(str)
            bottom5['Label'] = bottom5.index.astype(str)
        
            # --- Chart Top 5 ---
            fig_top, ax_top = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Label', y=feature, data=top5, color='steelblue', ax=ax_top)
            ax_top.set_title(f'Top 5 {feature}', fontsize=12)
            ax_top.set_xlabel('Index')
            ax_top.set_ylabel(feature)
            ax_top.tick_params(axis='x', rotation=45)
            st.pyplot(fig_top)
            plt.clf()

            # --- Chart Bottom 5 ---
            fig_bottom, ax_bottom = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Label', y=feature, data=bottom5, color='steelblue', ax=ax_bottom)
            ax_bottom.set_title(f'Bottom 5 {feature}', fontsize=12)
            ax_bottom.set_xlabel('Index')
            ax_bottom.set_ylabel(feature)
            ax_bottom.tick_params(axis='x', rotation=45)
            st.pyplot(fig_bottom)
            plt.clf()
        
        # Evaluasi Klaster
        st.subheader(translate("Evaluasi Klaster"))
        if "ANOVA" in cluster_evaluation_options:
            anova_results = perform_anova(df, selected_features)
            st.write(anova_results)
            interpret = ("\U0001F4CC Interpretasi Anova: P-value kurang dari alpha menunjukkan terdapat perbedaan signifikan." if language == "Indonesia"
                         else "\U0001F4CC ANOVA Interpretation: P-value less than alpha indicates significant difference.")
            st.write(interpret if (anova_results["P-Value"] < 0.05).any() else interpret.replace("kurang", "lebih").replace("terdapat", "tidak terdapat"))

        if "Silhouette Score" in cluster_evaluation_options:
            score = silhouette_score(df_scaled, df['KMeans_Cluster'])
            st.write(f"Silhouette Score: {score:.4f}")
            if score < 0:
                msg = "Silhouette Score rendah: klaster kurang baik."
            elif score > 0.5:
                msg = "Silhouette Score tinggi: klaster cukup baik."
            else:
                msg = "Silhouette Score sedang: kualitas klaster sedang."
            st.write("\U0001F4CC " + (msg if language == "Indonesia" else f"\U0001F4CC Silhouette Score Interpretation: {msg}"))

        if "Dunn Index" in cluster_evaluation_options:
            score = dunn_index(df_scaled.to_numpy(), df['KMeans_Cluster'].to_numpy())
            st.write(f"Dunn Index: {score:.4f}")
            msg = ("Dunn Index tinggi: pemisahan antar klaster baik." if score > 1
                   else "Dunn Index rendah: klaster saling tumpang tindih.")
            st.write("\U0001F4CC " + (msg if language == "Indonesia" else f"\U0001F4CC Dunn Index Interpretation: {msg}"))

else:
    st.warning("\u26A0 Silakan upload file Excel terlebih dahulu.")
