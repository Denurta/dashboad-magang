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
import io
import requests # Import library requests

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
        try:
            df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip()
            if 'Row Labels' not in df.columns:
                st.error("Kolom 'Row Labels' tidak ditemukan dalam file Excel. Fitur hapus berdasarkan nama baris tidak akan berfungsi.")
            st.session_state['df_original'] = df  # Simpan data asli
            st.session_state['df_cleaned'] = df.copy() # Inisialisasi df_cleaned
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
    st.info("\U0001F4CC Titik elbow terbaik adalah pada jumlah klaster di mana penurunan inertia mulai melambat secara signifikan. Titik ini menunjukkan jumlah klaster optimal.")

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
        "Download Template Excel": {"Indonesia": "Download Template Excel", "English": "Download Excel Template"},
        "Kolom 'Row Labels' tidak ditemukan dalam file Excel. Fitur hapus berdasarkan nama baris tidak akan berfungsi.": {
            "Indonesia": "Kolom 'Row Labels' tidak ditemukan dalam file Excel. Fitur hapus berdasarkan nama baris tidak akan berfungsi.",
            "English": "The 'Row Labels' column was not found in the Excel file. The remove rows by name feature will not work."
        },
        "Terjadi kesalahan saat membaca file": {"Indonesia": "Terjadi kesalahan saat membaca file", "English": "An error occurred while reading the file"},
        "Upload file Excel": {"Indonesia": "Upload file Excel", "English": "Upload Excel file"},
        "Kolom 'Row Labels' tidak ditemukan dalam data. Fitur hapus berdasarkan nama baris tidak akan berfungsi.": {
            "Indonesia": "Kolom 'Row Labels' tidak ditemukan dalam data. Fitur hapus berdasarkan nama baris tidak akan berfungsi.",
            "English": "The 'Row Labels' column was not found in the data. The remove rows by name feature will not work."
        },
        "Berhasil menghapus": {"Indonesia": "Berhasil menghapus", "English": "Successfully removed"},
        "baris dengan nama": {"Indonesia": "baris dengan nama", "English": "rows with names"},
        "Tidak ada baris dengan nama tersebut yang ditemukan.": {"Indonesia": "Tidak ada baris dengan nama tersebut yang ditemukan.", "English": "No rows with those names were found."},
        "Terjadi kesalahan saat menghapus baris": {"Indonesia": "Terjadi kesalahan saat menghapus baris", "English": "An error occurred while deleting rows"},
        "Pilih variabel untuk Elbow Method": {"Indonesia": "Pilih variabel untuk Elbow Method", "English": "Select variables for Elbow Method"},
        "Heatmap Korelasi Antar Fitur": {"Indonesia": "Heatmap Korelasi Antar Fitur", "English": "Feature Correlation Heatmap"},
        "Boxplot": {"Indonesia": "Boxplot", "English": "Boxplot"},
        "per Cluster": {"Indonesia": "per Cluster", "English": "per Cluster"},
        "Cluster": {"Indonesia": "Cluster", "English": "Cluster"},
        "Top 5 Terminal": {"Indonesia": "Top 5 Terminal", "English": "Top 5 Terminals"},
        "Bottom 5 Terminal": {"Indonesia": "Bottom 5 Terminal", "English": "Bottom 5 Terminals"},
        "Kolom 'Row Labels' tidak ditemukan pada data.": {"Indonesia": "Kolom 'Row Labels' tidak ditemukan pada data.", "English": "The 'Row Labels' column was not found in the data."},
        "\U0001F4CC Interpretasi Anova: P-value kurang dari alpha menunjukkan terdapat perbedaan signifikan.": {
            "Indonesia": "\U0001F4CC Interpretasi Anova: P-value kurang dari alpha menunjukkan terdapat perbedaan signifikan.",
            "English": "\U0001F4CC ANOVA Interpretation: P-value less than alpha indicates a significant difference."
        },
        "\U0001F4CC ANOVA Interpretation: P-value less than alpha indicates significant difference.": {
            "Indonesia": "\U0001F4CC Interpretasi Anova: P-value kurang dari alpha menunjukkan terdapat perbedaan signifikan.",
            "English": "\U0001F4CC ANOVA Interpretation: P-value less than alpha indicates a significant difference."
        },
        "*Silhouette Score*": {"Indonesia": "*Silhouette Score*", "English": "*Silhouette Score*"},
        "Silhouette Score rendah: klaster kurang baik.": {"Indonesia": "Silhouette Score rendah: klaster kurang baik.", "English": "Silhouette Score is low: poor clustering."},
        "Silhouette Score sedang: kualitas klaster sedang.": {"Indonesia": "Silhouette Score sedang: kualitas klaster sedang.", "English": "Silhouette Score is moderate: medium quality clustering."},
        "Silhouette Score tinggi: klaster cukup baik.": {"Indonesia": "Silhouette Score tinggi: klaster cukup baik.", "English": "Silhouette Score is high: good clustering."},
        "Silhouette Score is low: poor clustering.": {"Indonesia": "Silhouette Score rendah: klaster kurang baik.", "English": "Silhouette Score is low: poor clustering."},
        "Silhouette Score is moderate: medium quality clustering.": {"Indonesia": "Silhouette Score sedang: kualitas klaster sedang.", "English": "Silhouette Score is moderate: medium quality clustering."},
        "Silhouette Score is high: good clustering.": {"Indonesia": "Silhouette Score tinggi: klaster cukup baik.", "English": "Silhouette Score is high: good clustering."},
        "*Dunn Index*": {"Indonesia": "*Dunn Index*", "English": "*Dunn Index*"},
        "Dunn Index tinggi: pemisahan antar klaster baik.": {"Indonesia": "Dunn Index tinggi: pemisahan antar klaster baik.", "English": "Dunn Index is high: good separation between clusters."},
        "Dunn Index rendah: klaster saling tumpang tindih.": {"Indonesia": "Dunn Index rendah: klaster saling tumpang tindih.", "English": "Dunn Index is low: clusters overlap."},
        "Dunn Index is high: good separation between clusters.": {"Indonesia": "Dunn Index tinggi: pemisahan antar klaster baik.", "English": "Dunn Index is high: good separation between clusters."},
        "Dunn Index is low: clusters overlap.": {"Indonesia": "Dunn Index rendah: klaster saling tumpang tindih.", "English": "Dunn Index is low: clusters overlap."},
        "Panduan Pengguna": {"Indonesia": "Panduan Pengguna", "English": "User Guide"},
        "Download Template Excel": {"Indonesia": "Download Template Excel", "English": "Download Excel Template"},
        "dengan klik tombol berikut. Sesuaikan periode waktunya dengan periode waktu data anda dan jangan merubah nama provinsi. Data yang dimasukkan merupakan data runtun waktu seperti data nilai produksi, harga komoditas, temperatur udara, curah hujan, dan lainnya selama beberapa periode waktu.":{
            "English": "by clicking the button below. Adjust the time period to match your data's time period and do not change the province names. The data entered is time-series data such as production value data, commodity prices, air temperature, rainfall, and others over several time periods."
        },
        "Download Template Excel": {
            "Indonesia": "Download Template Excel",
            "English": "Download Excel Template"
        },
        "Kinerja Operasional Terminal": {
            "Indonesia": "Kinerja Operasional Terminal",
            "English": "Terminal Operational Performance"
        }
    }
    return translations.get(text, {}).get(language, text)

# --- Sidebar ---
with st.sidebar:
    st.subheader(translate("Jumlah Klaster"))
    n_clusters = st.slider("", 2, 10, 3)
    st.subheader(translate("Pilih Visualisasi"))
    visualization_options = st.multiselect("", ["Heatmap", "Boxplot", "Barchart"])
    st.subheader(translate("Pilih Evaluasi Klaster"))
    cluster_evaluation_options = st.multiselect("", ["ANOVA", "Silhouette Score", "Dunn Index"])
    st.subheader(translate("Hapus Baris"))
    drop_names = st.text_area(translate("Masukkan nama baris yang akan dihapus (pisahkan dengan koma)"), key="drop_names")
    drop_button = st.button(translate("Hapus Baris"))

    st.markdown("---")
    st.subheader(translate("Download Template Excel"))
    excel_template = pd.DataFrame({'Row Labels': ['Terminal A', 'Terminal B', 'Terminal C'],
                                   'Kinerja Operasional 1': [10, 15, 12],
                                   'Kinerja Operasional 2': [25, 30, 28],
                                   'Kinerja Operasional 3': [5, 8, 6]})
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        excel_template.to_excel(writer, sheet_name='Sheet1', index=False)
        writer.close()
    st.download_button(
        label="Download Template",
        data=buffer.getvalue(),
        file_name="terminal_clustering_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- Tampilan Utama ---
st.title(translate("Analisis Klaster Terminal"))

# Tambahkan bagian Panduan Pengguna
st.subheader(translate("Panduan Pengguna"))
st.markdown(f"""
    <div style="text-align: justify;">
    1. {translate("dengan klik tombol berikut. Sesuaikan periode waktunya dengan periode waktu data anda dan jangan merubah nama provinsi. Data yang dimasukkan merupakan data runtun waktu seperti data nilai produksi, harga komoditas, temperatur udara, curah hujan, dan lainnya selama beberapa periode waktu.")}
    </div>
    """, unsafe_allow_html=True)

# Tambahkan tombol untuk mengunduh template Excel
excel_template = pd.DataFrame({'Row Labels': ['Terminal A', 'Terminal B', 'Terminal C'],
                               'Kinerja Operasional 1': [10, 15, 12],
                               'Kinerja Operasional 2': [25, 30, 28],
                               'Kinerja Operasional 3': [5, 8, 6]})
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    excel_template.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()
st.download_button(
    label=translate("Download Template Excel"),
    data=buffer.getvalue(),
    file_name="terminal_clustering_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Area untuk upload data
uploaded_file = st.file_uploader(translate("Upload file Excel"), type=["xlsx"])
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()
        if 'Row Labels' not in df.columns:
            st.error(translate("Kolom 'Row Labels' tidak ditemukan dalam file Excel. Fitur hapus berdasarkan nama baris tidak akan berfungsi."))
        st.session_state['df_original'] = df  # Simpan data asli
        st.session_state['df_cleaned'] = df.copy() # Inisialisasi df_cleaned
        st.session_state['data_uploaded'] = True
    except Exception as e:
        st.error(f"{translate('Terjadi kesalahan saat membaca file')}: {e}")
else:
    st.info("⚠️ " + translate("Upload Data untuk Analisis"))

if 'data_uploaded' in st.session_state and st.session_state['data_uploaded']:
    df_cleaned = st.session_state['df_cleaned']

    if 'Row Labels' not in df_cleaned.columns:
        st.error(translate("Kolom 'Row Labels' tidak ditemukan dalam data. Fitur hapus berdasarkan nama baris tidak akan berfungsi."))
    else:
        if drop_button and drop_names:
            try:
                names_to_drop = [name.strip() for name in drop_names.split(',') if name.strip()]
                initial_rows = df_cleaned.shape[0]
                df_cleaned = df_cleaned[~df_cleaned['Row Labels'].isin(names_to_drop)]
                df_cleaned.reset_index(drop=True, inplace=True)
                st.session_state['df_cleaned'] = df_cleaned # Pastikan state df_cleaned diperbarui
                rows_deleted = initial_rows - df_cleaned.shape[0]
                if rows_deleted > 0:
                    st.success(f"✅ {translate('Berhasil menghapus')} {rows_deleted} {translate('baris dengan nama')}: {names_to_drop}")
                else:
                    st.info(translate("Tidak ada baris dengan nama tersebut yang ditemukan."))
            except Exception as e:
                st.error(f"❌ {translate('Terjadi kesalahan saat menghapus baris')}: {e}")

    # Gunakan df_cleaned yang ada di session state untuk analisis
    if 'df_cleaned' in st.session_state:
        df_cleaned_for_analysis = st.session_state['df_cleaned']
        # Ubah variabel features menjadi kinerja operasional terminal
        features = [col for col in df_cleaned_for_analysis.columns if 'Kinerja Operasional' in col]

        st.subheader(translate("Statistik Deskriptif"))
        st.dataframe(df_cleaned_for_analysis.describe())

        selected_features = st.multiselect(translate("Pilih variabel untuk Elbow Method"), features, default=features)

        if selected_features:
            df_scaled = normalize_data(df_cleaned_for_analysis, selected_features)

            st.subheader(translate("Metode Elbow"))
            elbow_method(df_scaled)

            df_cleaned_for_analysis['KMeans_Cluster'], kmeans_model = perform_kmeans(df_scaled, n_clusters)

            st.subheader(translate("Visualisasi Klaster"))

            if "Heatmap" in visualization_options:
                plt.figure(figsize=(10, 6))
                sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm')
                plt.title(translate("Heatmap Korelasi Antar Fitur"))
                st.pyplot(plt.gcf())
                plt.clf()

            if "Boxplot" in visualization_options:
                num_features = len(selected_features)
                fig, axes = plt.subplots(1, num_features, figsize=(5 * num_features, 5))
                if num_features == 1:
                    axes = [axes]
                for i, feature in enumerate(selected_features):
                    sns.boxplot(x='KMeans_Cluster', y=feature, data=df_cleaned_for_analysis, ax=axes[i])
                    axes[i].set_title(f"{translate('Boxplot')}: {feature} {translate('per Cluster')}")
                    axes[i].set_xlabel(translate("Cluster"))
                    axes[i].set_ylabel(feature)
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
                            fig_top, ax_top = plt.subplots(figsize=(4, 3))
                            sns.barplot(x=feature, y='Row Labels', data=top5, palette='Blues_d', ax=ax_top)
                            ax_top.set_title(f"{translate('Top 5 Terminal')} - {feature}")
                            st.pyplot(fig_top)
                            plt.clf()

                        with col2:
                            fig_bottom, ax_bottom = plt.subplots(figsize=(4, 3))
                            sns.barplot(x=feature, y='Row Labels', data=bottom5, palette='Blues_d', ax=ax_bottom)
                            ax_bottom.set_title(f"{translate('Bottom 5 Terminal')} - {feature}")
                            st.pyplot(fig_bottom)
                            plt.clf()
                else:
                    st.warning(translate("Kolom 'Row Labels' tidak ditemukan pada data."))

            st.subheader(translate("Evaluasi Klaster"))

            if "ANOVA" in cluster_evaluation_options:
                anova_results = perform_anova(df_cleaned_for_analysis, selected_features)
                st.write(anova_results)
                interpret = ("\U0001F4CC Interpretasi Anova: P-value kurang dari alpha menunjukkan terdapat perbedaan signifikan." if language == "Indonesia"
                             else "\U0001F4CC ANOVA Interpretation: P-value less than alpha indicates significant difference.")
                st.write(interpret if (anova_results["P-Value"] < 0.05).any() else interpret.replace("kurang", "lebih").replace("terdapat", "tidak terdapat"))

            if "Silhouette Score" in cluster_evaluation_options:
                score = silhouette_score(normalize_data(df_cleaned_for_analysis, selected_features), df_cleaned_for_analysis['KMeans_Cluster'])
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

            if "Dunn Index" in cluster_evaluation_options:
                score = dunn_index(normalize_data(df_cleaned_for_analysis, selected_features).to_numpy(), df_cleaned_for_analysis['KMeans_Cluster'].to_numpy())
                st.write(f"*Dunn Index*: {score:.4f}")

                # Pesan untuk Bahasa Indonesia
                msg_id = "Dunn Index tinggi: pemisahan antar klaster baik." if score > 1 else "Dunn Index rendah: klaster saling tumpang tindih."

                # Pesan untuk Bahasa Inggris
                msg_en = "Dunn Index is high: good separation between clusters." if score > 1 else "Dunn Index is low: clusters overlap."

                # Menampilkan pesan sesuai pilihan bahasa
                st.write("\U0001F4CC " + (msg_id if language == "Indonesia" else msg_en))
