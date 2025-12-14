# =====================================================
# FINAL DATA MINING
# Nama  : Nur Faizah Az Zahrah
# Judul : Analisis Segmentasi Penjualan Tiket Pesawat
# Metode: Agglomerative Clustering & Ensemble Regression
# =====================================================

# ========================
# IMPORT LIBRARY
# ========================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# ========================
# KONFIGURASI STREAMLIT
# ========================
st.set_page_config(page_title="UAS Data Mining", layout="wide")
st.title("Segmentasi Penjualan Tiket Pesawat")
st.write("Agglomerative Clustering & Ensemble Regression")

# ========================
# UPLOAD DATA
# ========================
uploaded_file = st.file_uploader(
    "Upload Dataset CSV Penjualan Tiket Pesawat",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Silakan upload file CSV untuk memulai analisis.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Data Awal")
st.dataframe(df.head())

# ========================
# 1. DATA CLEANING
# ========================
st.subheader("Data Cleaning")

st.write("Ukuran data awal:", df.shape)
st.write("Missing value per kolom:")
st.write(df.isnull().sum())

jumlah_duplikat = df.duplicated().sum()
st.write("Jumlah data duplikat sebelum cleaning:", jumlah_duplikat)

df = df.drop_duplicates()
st.write("Jumlah data duplikat setelah cleaning:", df.duplicated().sum())

# Konversi Date ke datetime
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

st.write("Ukuran data setelah cleaning:", df.shape)

# Contoh hasil konversi Date
if "Date" in df.columns:
    st.subheader("Contoh Kolom Date setelah Konversi")
    tampil_date = df[["Date"]].copy()
    tampil_date["Date"] = tampil_date["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(tampil_date.head())

# ========================
# 2. ENCODING DATA KATEGORIK
# ========================
st.subheader("Encoding Data Kategorik")

df_encoded = df.copy()
encoder = LabelEncoder()

for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col] = encoder.fit_transform(df_encoded[col])

st.write("Contoh 5 baris data setelah encoding:")
st.dataframe(df_encoded.head())

st.download_button(
    "Download Data Encoding",
    data=df_encoded.to_csv(index=False),
    file_name="data_encoded.csv",
    mime="text/csv"
)

# ========================
# 3. SCALING DATA
# ========================
st.subheader("Scaling Data")

datetime_cols = df_encoded.select_dtypes(include=["datetime64[ns]"]).columns
df_numeric = df_encoded.drop(columns=datetime_cols)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

st.write("Scaling menggunakan StandardScaler berhasil.")

# ========================
# 4. CLUSTERING (AGGLOMERATIVE)
# ========================
st.subheader("Clustering (Agglomerative Clustering)")

n_cluster = st.slider("Pilih jumlah cluster", 2, 5, 3)

cluster_model = AgglomerativeClustering(n_clusters=n_cluster)
df["Cluster"] = cluster_model.fit_predict(X_scaled)

st.write("Distribusi Data per Cluster:")
st.write(df["Cluster"].value_counts())

# ========================
# 5. PCA UNTUK VISUALISASI CLUSTER
# ========================
st.subheader("Visualisasi Clustering dengan PCA")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

fig_pca, ax_pca = plt.subplots()
sns.scatterplot(
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    data=df,
    palette="Set2",
    ax=ax_pca
)
ax_pca.set_title("Visualisasi Cluster (PCA)")
st.pyplot(fig_pca)

# Download visualisasi PCA
buf = io.BytesIO()
fig_pca.savefig(buf, format="png", bbox_inches="tight")
buf.seek(0)

st.download_button(
    "Download Visualisasi PCA",
    data=buf,
    file_name="visualisasi_cluster_pca.png",
    mime="image/png"
)

# ========================
# 6. FEATURE ENGINEERING DARI DATE
# ========================
if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
    st.subheader("Feature Engineering Berbasis Waktu")

    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots()
        sns.countplot(x="Month", data=df, ax=ax)
        ax.set_title("Distribusi Transaksi per Bulan")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(x="Day", data=df, ax=ax)
        ax.set_title("Distribusi Transaksi per Hari")
        st.pyplot(fig)

    with col3:
        fig, ax = plt.subplots()
        sns.countplot(x="DayOfWeek", data=df, ax=ax)
        ax.set_title("Distribusi Transaksi per Hari (0=Senin)")
        st.pyplot(fig)

# ========================
# 7. ENSEMBLE REGRESSION
# ========================
st.subheader("Ensemble Regression per Cluster")

target_column = "Ticket_Quantity"
ensemble_results = []

for cluster in sorted(df["Cluster"].unique()):
    data_cluster = df[df["Cluster"] == cluster]

    if len(data_cluster) < 10:
        continue

    X = data_cluster.drop(
        columns=[target_column, "Cluster", "PCA1", "PCA2"],
        errors="ignore"
    )
    y = data_cluster[target_column]

    X = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.01)
    elastic = ElasticNet(alpha=0.01, l1_ratio=0.5)

    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)
    elastic.fit(X_train, y_train)

    y_pred = (
        ridge.predict(X_test)
        + lasso.predict(X_test)
        + elastic.predict(X_test)
    ) / 3

    ensemble_results.append({
        "Cluster": cluster,
        "Jumlah_Data": len(data_cluster),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2_Score": r2_score(y_test, y_pred)
    })

ensemble_df = pd.DataFrame(ensemble_results)
st.dataframe(ensemble_df)

# ========================
# 8. DOWNLOAD OUTPUT AKHIR
# ========================
st.subheader("Download Output Akhir")

st.download_button(
    "Download Data Final (Clustering + PCA)",
    data=df.to_csv(index=False),
    file_name="hasil_clustering_final.csv",
    mime="text/csv"
)
