# --- Import Library ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway
from scipy.spatial.distance import pdist, squareform

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

# --- Fungsi Utama ---
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
    labels = kmeans.fit_predict(df_scaled)
    return labels, kmeans

def elbow_method(df_scaled):
    distortions = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        distortions.append(kmeans.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), distortions, marker='o', color='steelblue')
    plt.title('Metode Elbow')
    plt.xlabel('Jumlah Klaster')
    plt.ylabel('Inertia')
    st.pyplot(plt.gcf())
    plt.clf()

def perform_anova(df, features):
    results = []
    for f in features:
        groups = [df[df['KMeans_Cluster'] == k][f] for k in df['KMeans_Cluster'].unique()]
        f_stat, p_val = f_oneway(*groups)
        results.append({"Variabel": f, "F-Stat": f_stat, "P-Value": p_val})
    return pd.DataFrame(results)

def dunn_index(df_scaled, labels):
    distances = squareform(pdist(df_scaled))
    unique_labels = np.unique(labels)
    intra = [np.max(pdist(df_scaled[labels == c])) for c in unique_labels if np.sum(labels == c) > 1]
    inter = [np.min(pdist(np.vstack((df_scaled[labels == i], df_scaled[labels == j]))))
             for i in unique_labels for j in unique_labels if i < j]
    return np.min(inter) / np.max(intra)

# --- Bahasa dan Terjemahan ---
lang = st.sidebar.radio("Pilih Bahasa", ["Indonesia", "English"])

def t(text):
    dicts = {
        "Jumlah Klaster": {"Indonesia": "Jumlah Klaster", "English": "Number of Clusters"},
        "Pilih Visualisasi": {"Indonesia": "Pilih Visualisasi", "English": "Select Visualization"},
        "Pilih Evaluasi Klaster": {"Indonesia": "Pilih Evaluasi Klaster", "English": "Select Cluster Evaluation"},
        "Analisis Klaster Terminal": {"Indonesia": "Analisis Klaster Terminal", "English": "Terminal Cluster Analysis"},
        "Statistik Deskriptif": {"Indonesia": "Statistik Deskriptif", "English": "Descriptive Statistics"},
        "Evaluasi Klaster": {"Indonesia": "Evaluasi Klaster", "English": "Cluster Evaluation"},
        "Metode Elbow": {"Indonesia": "Metode Elbow", "English": "Elbow Method"},
        "Visualisasi Klaster": {"Indonesia": "Visualisasi Klaster", "English": "Cluster Visualization"},
    }
    return dicts.get(text, {}).get(lang, text)

# --- Sidebar ---
st.sidebar.title("\u26f4 Clustering Terminal")
n_cluster = st.sidebar.slider(t("Jumlah Klaster"), 2, 10, 3)
vis_options = st.sidebar.multiselect(t("Pilih Visualisasi"), ["Scatter Plot", "Heatmap", "Boxplot", "Barchart"])
eval_options = st.sidebar.multiselect(t("Pilih Evaluasi Klaster"), ["ANOVA", "Silhouette Score", "Dunn Index"])

# --- Main App ---
st.title(t("Analisis Klaster Terminal"))
df = load_data()

if df is not None:
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    st.subheader(t("Statistik Deskriptif"))
    st.write(df.describe())

    selected_features = st.multiselect("Pilih variabel", numeric_cols, default=numeric_cols)

    if selected_features:
        df_scaled = normalize_data(df, selected_features)

        st.subheader(t("Metode Elbow"))
        elbow_method(df_scaled)

        df['KMeans_Cluster'], model = perform_kmeans(df_scaled, n_cluster)

        st.subheader(t("Visualisasi Klaster"))
        if "Scatter Plot" in vis_options:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=df_scaled[selected_features[0]], y=df_scaled[selected_features[1]], hue=df['KMeans_Cluster'], palette='viridis')
            plt.title(f"Scatter: {selected_features[0]} vs {selected_features[1]}")
            st.pyplot(plt.gcf())
            plt.clf()

        if "Heatmap" in vis_options:
            plt.figure(figsize=(8, 5))
            sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt.gcf())
            plt.clf()

        if "Boxplot" in vis_options:
            fig, axes = plt.subplots(1, len(selected_features), figsize=(5*len(selected_features), 5))
            if len(selected_features) == 1: axes = [axes]
            for i, f in enumerate(selected_features):
                sns.boxplot(x='KMeans_Cluster', y=f, data=df, ax=axes[i])
                axes[i].set_title(f)
            st.pyplot(fig)
            plt.clf()

        if "Barchart" in vis_options and 'Row Labels' in df.columns:
            for f in selected_features:
                grouped = df.groupby('Row Labels')[f].mean().reset_index()
                top5 = grouped.nlargest(5, f)
                bottom5 = grouped.nsmallest(5, f)
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.barplot(x=f, y='Row Labels', data=top5, palette='Blues_d', ax=ax)
                    ax.set_title(f'Top 5 {f}')
                    st.pyplot(fig)
                    plt.clf()
                with col2:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.barplot(x=f, y='Row Labels', data=bottom5, palette='Blues_d', ax=ax)
                    ax.set_title(f'Bottom 5 {f}')
                    st.pyplot(fig)
                    plt.clf()

        st.subheader(t("Evaluasi Klaster"))
        if "ANOVA" in eval_options:
            anova = perform_anova(df, selected_features)
            st.write(anova)
        if "Silhouette Score" in eval_options:
            score = silhouette_score(df_scaled, df['KMeans_Cluster'])
            st.write(f"Silhouette Score: {score:.4f}")
        if "Dunn Index" in eval_options:
            dunn = dunn_index(df_scaled.to_numpy(), df['KMeans_Cluster'].to_numpy())
            st.write(f"Dunn Index: {dunn:.4f}")
else:
    st.warning("\u26A0 Silakan upload file Excel terlebih dahulu.")
