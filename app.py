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
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Styling CSS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, rgba(135, 206, 250, 0.4), rgba(70, 130, 180, 0.4));
        color: #1E3A5F;
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
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features, index=df.index)
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

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K, distortions, marker='o', linestyle='-')
    ax.set_xlabel('Jumlah Klaster')
    ax.set_ylabel('Inertia')
    ax.set_title('Metode Elbow')
    st.pyplot(fig)
    plt.close(fig)
    st.info("Tips: Titik elbow terbaik adalah pada jumlah klaster di mana penurunan inertia mulai melambat secara signifikan.")

def perform_anova(df, features):
    results = []
    for feature in features:
        groups = [df[df['KMeans_Cluster'] == k][feature] for k in df['KMeans_Cluster'].unique()]
        if all(len(g) > 1 for g in groups):
            f_stat, p_val = f_oneway(*groups)
            results.append({"Variabel": feature, "F-Stat": f_stat, "P-Value": p_val})
        else:
            results.append({"Variabel": feature, "F-Stat": None, "P-Value": None})
    return pd.DataFrame(results)

def dunn_index(df_scaled, labels):
    distances = squareform(pdist(df_scaled, metric='euclidean'))
    unique_clusters = np.unique(labels)
    intra = []
    inter = []

    for cluster in unique_clusters:
        points = df_scaled[labels == cluster]
        if len(points) > 1:
            intra.append(np.max(pdist(points)))
    for i in range(len(unique_clusters)):
        for j in range(i + 1, len(unique_clusters)):
            c_i = df_scaled[labels == unique_clusters[i]]
            c_j = df_scaled[labels == unique_clusters[j]]
            inter.append(np.min(pdist(np.vstack((c_i, c_j)))))

    if intra and inter:
        return np.min(inter) / np.max(intra)
    return 0

# --- Sidebar & Bahasa ---
st.sidebar.title("Clustering Terminal")
language = st.sidebar.radio("Pilih Bahasa", ["Indonesia", "English"])

def translate(text):
    translations = {
        "Jumlah Klaster": {"Indonesia": "Jumlah Klaster", "English": "Number of Clusters"},
        "Pilih Visualisasi": {"Indonesia": "Pilih Visualisasi", "English": "Select Visualization"},
        "Pilih Evaluasi Klaster": {"Indonesia": "Pilih Evaluasi Klaster", "English": "Select Cluster Evaluation"},
        "Hapus Baris": {"Indonesia": "Hapus Baris", "English": "Remove Rows"},
        "Masukkan indeks baris yang akan dihapus (pisahkan dengan koma)": {
            "Indonesia": "Masukkan indeks baris yang akan dihapus (pisahkan dengan koma)",
            "English": "Enter row indices to remove (separate with commas)"
        },
        "Analisis Klaster Terminal": {"Indonesia": "Analisis Klaster Terminal", "English": "Terminal Cluster Analysis"},
        "Metode Elbow": {"Indonesia": "Metode Elbow", "English": "Elbow Method"},
        "Visualisasi Klaster": {"Indonesia": "Visualisasi Klaster", "English": "Cluster Visualization"},
        "Statistik Deskriptif": {"Indonesia": "Statistik Deskriptif", "English": "Descriptive Statistics"},
        "Evaluasi Klaster": {"Indonesia": "Evaluasi Klaster", "English": "Cluster Evaluation"},
    }
    return translations.get(text, {}).get(language, text)

n_clusters = st.sidebar.slider(translate("Jumlah Klaster"), 2, 10, 3)
visualization_options = st.sidebar.multiselect(translate("Pilih Visualisasi"), ["Heatmap", "Boxplot", "Barchart"])
cluster_evaluation_options = st.sidebar.multiselect(translate("Pilih Evaluasi Klaster"), ["ANOVA", "Silhouette Score", "Dunn Index"])
st.sidebar.subheader(translate("Hapus Baris"))
drop_rows = st.sidebar.text_area(translate("Masukkan indeks baris yang akan dihapus (pisahkan dengan koma)"))

# --- Tampilan Utama ---
st.title(translate("Analisis Klaster Terminal"))
df = load_data()

if df is not None:
    if drop_rows:
        try:
            drop_indices = [int(i.strip()) for i in drop_rows.split(',') if i.strip().isdigit()]
            df = df.drop(index=drop_indices, errors='ignore').reset_index(drop=True)
            st.success(f"Berhasil menghapus baris: {drop_indices}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat menghapus baris: {e}")

    features = df.select_dtypes(include='number').columns.tolist()
    if features:
        st.subheader(translate("Statistik Deskriptif"))
        st.dataframe(df.describe())

        selected_features = st.multiselect("Pilih variabel untuk Elbow Method", features, default=features)
        if selected_features:
            df_scaled = normalize_data(df, selected_features)
            st.subheader(translate("Metode Elbow"))
            elbow_method(df_scaled)

            df['KMeans_Cluster'], kmeans_model = perform_kmeans(df_scaled, n_clusters)

            # --- Visualisasi Klaster ---
            st.subheader(translate("Visualisasi Klaster"))

            if "Heatmap" in visualization_options:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm', ax=ax)
                ax.set_title("Heatmap Korelasi Antar Fitur")
                st.pyplot(fig)
                plt.close(fig)

            if "Boxplot" in visualization_options:
                fig, axes = plt.subplots(1, len(selected_features), figsize=(5 * len(selected_features), 5))
                if len(selected_features) == 1:
                    axes = [axes]
                for i, feature in enumerate(selected_features):
                    sns.boxplot(x='KMeans_Cluster', y=feature, data=df, ax=axes[i])
                    axes[i].set_title(f"{feature} per Cluster")
                st.pyplot(fig)
                plt.close(fig)

            if "Barchart" in visualization_options:
                if 'Row Labels' in df.columns:
                    for feature in selected_features:
                        grouped = df.groupby('Row Labels')[feature].mean().reset_index()
                        top5 = grouped.nlargest(5, feature)
                        bottom5 = grouped.nsmallest(5, feature)
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_top, ax = plt.subplots()
                            sns.barplot(data=top5, x=feature, y='Row Labels', ax=ax)
                            ax.set_title(f"Top 5 Terminal: {feature}")
                            st.pyplot(fig_top)
                            if feature == "ET/BT":
                                st.info("Terminal yang masuk Top 5 untuk variabel ET/BT dapat dikategorikan sebagai Klaster Efisien.")
                        with col2:
                            fig_bot, ax = plt.subplots()
                            sns.barplot(data=bottom5, x=feature, y='Row Labels', ax=ax)
                            ax.set_title(f"Bottom 5 Terminal: {feature}")
                            st.pyplot(fig_bot)
                            if feature in ["BT", "BWT"]:
                                st.warning("Terminal yang masuk Bottom 5 untuk variabel BT dan BWT cenderung tidak efisien.")
                else:
                    st.warning("Kolom 'Row Labels' tidak tersedia.")

            # --- Evaluasi Klaster ---
st.subheader(translate("Evaluasi Klaster"))

if "ANOVA" in cluster_evaluation_options:
    st.markdown("📌 **Hasil ANOVA**")
    anova_df = perform_anova(df, selected_features)
    st.dataframe(anova_df)
    has_significant = (anova_df["P-Value"] < 0.05).any()
    interpretasi = (
        "📌 **Interpretasi ANOVA:** Terdapat perbedaan signifikan antar klaster berdasarkan beberapa variabel (p < 0.05)." 
        if has_significant else 
        "📌 **Interpretasi ANOVA:** Tidak ditemukan perbedaan signifikan antar klaster untuk variabel-variabel tersebut (p ≥ 0.05)."
    )
    st.markdown(interpretasi)

if "Silhouette Score" in cluster_evaluation_options:
    sil_score = silhouette_score(df_scaled, df['KMeans_Cluster'])
    st.markdown(f"📌 **Silhouette Score**: {sil_score:.4f}")
    level = ("rendah" if sil_score < 0.25 else "sedang" if sil_score <= 0.5 else "tinggi")
    st.markdown(f"📌 **Interpretasi Silhouette Score:** Kualitas klaster **{level}**.")

if "Dunn Index" in cluster_evaluation_options:
    dunn_score = dunn_index(df_scaled.to_numpy(), df['KMeans_Cluster'].to_numpy())
    st.markdown(f"📌 **Dunn Index**: {dunn_score:.4f}")
    interpretasi_dunn = (
        "📌 **Interpretasi Dunn Index:** Nilai Dunn Index tinggi: pemisahan antar klaster **baik**."
        if dunn_score > 1 else 
        "📌 **Interpretasi Dunn Index:** Nilai Dunn Index rendah: klaster cenderung **saling tumpang tindih**."
    )
    st.markdown(interpretasi_dunn)
