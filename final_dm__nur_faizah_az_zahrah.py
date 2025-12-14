# ===============================
# IMPORT LIBRARY
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score


# ===============================
# CONFIG STREAMLIT
# ===============================
st.set_page_config(page_title="UAS Data Mining", layout="wide")
st.title("Analisis Clustering dan Regresi Ensemble")


# ===============================
# LOAD DATA
# ===============================
uploaded_file = st.file_uploader(
    "Upload dataset CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Silakan upload file CSV untuk memulai analisis")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Data Awal")
st.dataframe(df.head())


# ===============================
# DATA CLEANING
# ===============================
st.header("🧹 Data Cleaning")

st.write("Ukuran data awal (baris, kolom):", df.shape)

# Missing value
st.subheader("Missing Value per Kolom")
missing_df = df.isnull().sum().reset_index()
missing_df.columns = ["Kolom", "Jumlah Missing"]
st.dataframe(missing_df)

# Duplikat
jumlah_duplikat = df.duplicated().sum()
st.write("Jumlah data duplikat sebelum dihapus:", jumlah_duplikat)

df = df.drop_duplicates()

st.write("Jumlah data duplikat setelah dihapus:", df.duplicated().sum())
st.write("Ukuran data setelah cleaning:", df.shape)

# Konversi Date
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    st.subheader("Contoh Kolom Date Setelah Konversi")
    st.dataframe(df[["Date"]].head())


# ===============================
# ENCODING
# ===============================
df_encoded = df.copy()
encoder = LabelEncoder()

for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col] = encoder.fit_transform(df_encoded[col])

st.subheader("📁 Data Setelah Encoding")
st.dataframe(df_encoded.head())

csv_encoded = df_encoded.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Data Encoding (CSV)",
    csv_encoded,
    "data_encoded.csv",
    "text/csv"
)


# ===============================
# FEATURE ENGINEERING (DATE)
# ===============================
if "Date" in df_encoded.columns:
    df_encoded["Month"] = df_encoded["Date"].dt.month
    df_encoded["Day"] = df_encoded["Date"].dt.day
    df_encoded["DayOfWeek"] = df_encoded["Date"].dt.dayofweek


# ===============================
# CLUSTERING DATA PREPARATION
# ===============================
fitur_clustering = [
    "Ticket_Quantity",
    "Total_Price",
    "Month",
    "DayOfWeek"
]

fitur_clustering = [c for c in fitur_clustering if c in df_encoded.columns]
st.write("Fitur clustering yang digunakan:", fitur_clustering)

X_cluster = df_encoded[fitur_clustering]

scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)


# =====================================================
# ANALISIS 1: CLUSTERING (AGGLOMERATIVE)
# =====================================================
st.header("🔵 Analisis Clustering (Agglomerative)")

cluster_model = AgglomerativeClustering(n_clusters=3)
df_encoded["Cluster"] = cluster_model.fit_predict(X_cluster_scaled)

st.write("Distribusi Cluster:")
st.write(df_encoded["Cluster"].value_counts())


# ===============================
# PCA CLUSTERING
# ===============================
pca_cluster = PCA(n_components=2)
X_pca_cluster = pca_cluster.fit_transform(X_cluster_scaled)

df_encoded["PCA1"] = X_pca_cluster[:, 0]
df_encoded["PCA2"] = X_pca_cluster[:, 1]

fig_c, ax_c = plt.subplots()
sns.scatterplot(
    data=df_encoded,
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette="Set2",
    ax=ax_c
)
ax_c.set_title("PCA Clustering (Agglomerative)")
st.pyplot(fig_c)

buf_c = io.BytesIO()
fig_c.savefig(buf_c, format="png", bbox_inches="tight")
buf_c.seek(0)
st.download_button(
    "Download Visualisasi PCA Clustering",
    buf_c,
    "pca_clustering.png",
    "image/png"
)


# ===============================
# EVALUASI CLUSTERING
# ===============================
st.subheader("Evaluasi Clustering (Silhouette Score)")

silhouette_vals = silhouette_samples(X_cluster_scaled, df_encoded["Cluster"])
df_encoded["Silhouette"] = silhouette_vals

cluster_eval = df_encoded.groupby("Cluster").agg(
    Jumlah_Data=("Cluster", "count"),
    Silhouette_Score=("Silhouette", "mean")
).reset_index()

st.dataframe(cluster_eval)

overall_silhouette = silhouette_score(X_cluster_scaled, df_encoded["Cluster"])

st.dataframe(pd.DataFrame({
    "Jumlah_Cluster": [df_encoded["Cluster"].nunique()],
    "Jumlah_Data": [len(df_encoded)],
    "Silhouette_Score_Keseluruhan": [overall_silhouette]
}))


# =====================================================
# ANALISIS 2: REGRESI ENSEMBLE
# =====================================================
st.header("🟢 Analisis Regresi Ensemble")

target_column = "Ticket_Quantity"

fitur_regresi = [
    "Total_Price",
    "Month",
    "Day",
    "DayOfWeek"
]

fitur_regresi += list(df_encoded.select_dtypes(include=[np.number]).columns)
fitur_regresi = list(set(fitur_regresi))
fitur_regresi = [c for c in fitur_regresi if c in df_encoded.columns and c != target_column]

st.write("Fitur regresi yang digunakan:", fitur_regresi)

X_regresi = df_encoded[fitur_regresi]
y_regresi = df_encoded[target_column]

scaler_reg = StandardScaler()
X_regresi_scaled = scaler_reg.fit_transform(X_regresi)

X_train, X_test, y_train, y_test = train_test_split(
    X_regresi_scaled, y_regresi, test_size=0.2, random_state=42
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

st.dataframe(pd.DataFrame([{
    "MSE": mean_squared_error(y_test, y_pred),
    "R2_Score": r2_score(y_test, y_pred),
    "Jumlah_Data": len(df_encoded)
}]))
