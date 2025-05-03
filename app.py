# --- Import Library ---

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import f\_oneway
from sklearn.metrics import silhouette\_score
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
} </style>
""", unsafe\_allow\_html=True)

# --- Fungsi ---

def load\_data():
uploaded\_file = st.file\_uploader("Upload file Excel", type=\["xlsx"])
if uploaded\_file is not None:
try:
df = pd.read\_excel(uploaded\_file)
df.columns = df.columns.str.strip()
if 'Row Labels' not in df.columns:
st.error("Kolom 'Row Labels' tidak ditemukan dalam file Excel. Fitur hapus berdasarkan nama baris tidak akan berfungsi.")
st.session\_state\['df\_original'] = df
st.session\_state\['df\_cleaned'] = df.copy()
st.session\_state\['data\_uploaded'] = True
return True
except Exception as e:
st.error(f"Terjadi kesalahan saat membaca file: {e}")
return False
return False

def normalize\_data(df, features):
scaler = StandardScaler()
df\_scaled = pd.DataFrame(scaler.fit\_transform(df\[features]), columns=features)
df\_scaled.index = df.index
return df\_scaled

def perform\_kmeans(df\_scaled, n\_clusters):
kmeans = KMeans(n\_clusters=n\_clusters, random\_state=42, n\_init=10)
clusters = kmeans.fit\_predict(df\_scaled)
return clusters, kmeans

def elbow\_method(df\_scaled):
distortions = \[]
K = range(1, 11)
for k in K:
kmeans = KMeans(n\_clusters=k, random\_state=42, n\_init=10)
kmeans.fit(df\_scaled)
distortions.append(kmeans.inertia\_)

```
plt.figure(figsize=(10, 6))
plt.plot(K, distortions, color='steelblue', marker='o', linestyle='-', markersize=8)
plt.xlabel('Jumlah Klaster')
plt.ylabel('Inertia')
plt.title('Metode Elbow')
st.pyplot(plt.gcf())
plt.clf()
st.info("\U0001F4CC Titik elbow terbaik adalah pada jumlah klaster di mana penurunan inertia mulai melambat secara signifikan.")
```

def perform\_anova(df, features):
anova\_results = \[]
for feature in features:
groups = \[df\[df\['KMeans\_Cluster'] == k]\[feature] for k in df\['KMeans\_Cluster'].unique()]
f\_stat, p\_value = f\_oneway(\*groups)
anova\_results.append({"Variabel": feature, "F-Stat": f\_stat, "P-Value": p\_value})
return pd.DataFrame(anova\_results)

def dunn\_index(df\_scaled, labels):
distances = squareform(pdist(df\_scaled, metric='euclidean'))
unique\_clusters = np.unique(labels)

```
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
```

# --- Sidebar & Bahasa ---

st.sidebar.title("\u26f4 Clustering Terminal")
language = st.sidebar.radio("Pilih Bahasa", \["Indonesia", "English"])

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
}
return translations.get(text, {}).get(language, text)

# --- Sidebar ---

with st.sidebar:
st.subheader(translate("Jumlah Klaster"))
n\_clusters = st.slider("", 2, 10, 3)
st.subheader(translate("Pilih Visualisasi"))
visualization\_options = st.multiselect("", \["Heatmap", "Boxplot", "Barchart"])
st.subheader(translate("Pilih Evaluasi Klaster"))
cluster\_evaluation\_options = st.multiselect("", \["ANOVA", "Silhouette Score", "Dunn Index"])
st.subheader(translate("Hapus Baris"))
drop\_names = st.text\_area(translate("Masukkan nama baris yang akan dihapus (pisahkan dengan koma)"), key="drop\_names")
drop\_button = st.button(translate("Hapus Baris"))

# --- Tampilan Utama ---

st.title(translate("Analisis Klaster Terminal"))

# Upload Data

data\_loaded = load\_data()
if not data\_loaded:
st.info("⚠️ " + translate("Upload Data untuk Analisis"))

if 'data\_uploaded' in st.session\_state and st.session\_state\['data\_uploaded']:
df\_cleaned = st.session\_state\['df\_cleaned']

```
if 'Row Labels' in df_cleaned.columns:
    if drop_button and drop_names:
        names_to_drop = [name.strip() for name in drop_names.split(',') if name.strip()]
        initial_rows = df_cleaned.shape[0]
        df_cleaned = df_cleaned[~df_cleaned['Row Labels'].isin(names_to_drop)]
        df_cleaned.reset_index(drop=True, inplace=True)
        st.session_state['df_cleaned'] = df_cleaned
        rows_deleted = initial_rows - df_cleaned.shape[0]
        if rows_deleted > 0:
            st.success(f"✅ Berhasil menghapus {rows_deleted} baris dengan nama: {names_to_drop}")
        else:
            st.info("Tidak ada baris dengan nama tersebut yang ditemukan.")
else:
    st.error("Kolom 'Row Labels' tidak ditemukan dalam data.")

if 'df_cleaned' in st.session_state:
    df_cleaned_for_analysis = st.session_state['df_cleaned']
    features = df_cleaned_for_analysis.select_dtypes(include='number').columns.tolist()

    st.subheader(translate("Statistik Deskriptif"))
    st.dataframe(df_cleaned_for_analysis.describe())

    selected_features = st.multiselect("Pilih variabel untuk Elbow Method", features, default=features)

    if selected_features:
        df_scaled = normalize_data(df_cleaned_for_analysis, selected_features)

        st.subheader(translate("Metode Elbow"))
        elbow_method(df_scaled)

        df_cleaned_for_analysis['KMeans_Cluster'], _ = perform_kmeans(df_scaled, n_clusters)

        st.subheader(translate("Visualisasi Klaster"))

        if "Heatmap" in visualization_options:
            plt.figure(figsize=(10, 6))
            sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm')
            plt.title("Heatmap Korelasi Antar Fitur")
            st.pyplot(plt.gcf())
            plt.clf()

        if "Boxplot" in visualization_options:
            num_features = len(selected_features)
            fig, axes = plt.subplots(1, num_features, figsize=(5 * num_features, 5))
            if num_features == 1:
                axes = [axes]
            for i, feature in enumerate(selected_features):
                sns.boxplot(x='KMeans_Cluster', y=feature, data=df_cleaned_for_analysis, ax=axes[i])
                axes[i].set_title(f"Boxplot: {feature} per Cluster")
                axes[i].set_xlabel("Cluster")
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
                        ax_top.set_title(f"Top 5 Terminal - {feature}")
                        st.pyplot(fig_top)
                        plt.clf()

                    with col2:
                        fig_bottom, ax_bottom = plt.subplots(figsize=(4, 3))
                        sns.barplot(x=feature, y='Row Labels', data=bottom5, palette='Blues_d', ax=ax_bottom)
                        ax_bottom.set_title(f"Bottom 5 Terminal - {feature}")
                        st.pyplot(fig_bottom)
                        plt.clf()
            else:
                st.warning("Kolom 'Row Labels' tidak ditemukan pada data.")

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
            msg_id = "Dunn Index tinggi: pemisahan antar klaster baik." if score > 1 else "Dunn Index rendah: klaster saling tumpang tindih."
            msg_en = "Dunn Index is high: good separation between clusters." if score > 1 else "Dunn Index is low: clusters overlap."
            st.write("\U0001F4CC " + (msg_id if language == "Indonesia" else msg_en))
```

# --- Panduan Penggunaan ---

with st.expander("ℹ️ Panduan Penggunaan Aplikasi" if language == "Indonesia" else "ℹ️ Application Usage Guide"):
if language == "Indonesia":
st.markdown("""
1\. **Upload File Excel:** Klik tombol "Browse files" untuk mengunggah file data Anda (format `.xlsx`).
2\. **Pilih Jumlah Klaster:** Tentukan jumlah klaster yang diinginkan menggunakan slider.
3\. **Hapus Baris (Opsional):** Masukkan nama terminal pada kolom 'Row Labels' yang ingin dihapus, pisahkan dengan koma.
4\. **Pilih Visualisasi & Evaluasi:** Centang visualisasi atau evaluasi klaster yang ingin ditampilkan.
5\. **Interpretasi:** Hasil akan ditampilkan secara otomatis setelah data dan parameter dimasukkan.
""")
else:
st.markdown("""
1\. **Upload Excel File:** Click "Browse files" to upload your data file (in `.xlsx` format).
2\. **Select Number of Clusters:** Use the slider to choose how many clusters you want.
3\. **Remove Rows (Optional):** Enter row names from the 'Row Labels' column to be removed, separated by commas.
4\. **Select Visualizations & Evaluations:** Check any cluster visualizations or evaluations you want to see.
5\. **Interpretation:** The results will be displayed automatically after data and parameters are provided.
""")
