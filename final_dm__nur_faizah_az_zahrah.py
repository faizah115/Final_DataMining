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

st.subheader("Missing Value per Kolom")
missing_df = df.isnull().sum().reset_index()
missing_df.columns = ["Kolom", "Jumlah Missing"]
st.dataframe(missing_df)

jumlah_duplikat = df.duplicated().sum()
st.write("Jumlah data duplikat sebelum dihapus:", jumlah_duplikat)

df = df.drop_duplicates()
st.write("Jumlah data duplikat setelah dihapus:", df.duplicated().sum())

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    st.subheader("Contoh Kolom Date Setelah Konversi")
    st.dataframe(df[["Date"]].head())

st.write("Ukuran data setelah cleaning:", df.shape)

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

# =====================================================
# FEATURE ENGINEERING KHUSUS CLUSTERING
# =====================================================
st.header("⚙️ Feature Engineering untuk Clustering")

df_cluster_fe = df_encoded.copy()

if "Total_Price" in df_cluster_fe.columns:
    df_cluster_fe["Log_Total_Price"] = np.log1p(df_cluster_fe["Total_Price"])

if "Ticket_Quantity" in df_cluster_fe.columns:
    df_cluster_fe["Log_Ticket_Quantity"] = np.log1p(df_cluster_fe["Ticket_Quantity"])

if set(["Total_Price", "Ticket_Quantity"]).issubset(df_cluster_fe.columns):
    df_cluster_fe["Avg_Price_per_Ticket"] = (
        df_cluster_fe["Total_Price"] / (df_cluster_fe["Ticket_Quantity"] + 1)
    )

fitur_clustering_fe = [
    "Log_Total_Price",
    "Log_Ticket_Quantity",
    "Avg_Price_per_Ticket",
    "Month",
    "DayOfWeek"
]

fitur_clustering_fe = [c for c in fitur_clustering_fe if c in df_cluster_fe.columns]
st.write("Fitur clustering:", fitur_clustering_fe)

X_cluster = df_cluster_fe[fitur_clustering_fe]
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# =====================================================
# CLUSTERING AGGLOMERATIVE
# =====================================================
st.header("🔵 Analisis Clustering (Agglomerative)")

cluster_model = AgglomerativeClustering(n_clusters=3)
df_encoded["Cluster"] = cluster_model.fit_predict(X_cluster_scaled)

st.write("Distribusi Cluster")
st.write(df_encoded["Cluster"].value_counts())

pca_cluster = PCA(n_components=2)
X_pca = pca_cluster.fit_transform(X_cluster_scaled)
df_encoded["PCA1"] = X_pca[:, 0]
df_encoded["PCA2"] = X_pca[:, 1]

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

buf = io.BytesIO()
fig_c.savefig(buf, format="png", bbox_inches="tight")
buf.seek(0)
st.download_button("Download PCA Clustering", buf, "pca_clustering.png", "image/png")

sil_vals = silhouette_samples(X_cluster_scaled, df_encoded["Cluster"])
df_encoded["Silhouette"] = sil_vals

cluster_eval = df_encoded.groupby("Cluster").agg(
    Jumlah_Data=("Cluster", "count"),
    Silhouette_Score=("Silhouette", "mean")
).reset_index()

st.dataframe(cluster_eval)

overall_sil = silhouette_score(X_cluster_scaled, df_encoded["Cluster"])
st.write("Silhouette Score Keseluruhan:", overall_sil)

# =====================================================
# FEATURE ENGINEERING KHUSUS REGRESI
# =====================================================
st.header("⚙️ Feature Engineering untuk Regresi")

df_reg_fe = df_encoded.copy()

if set(["Total_Price", "Ticket_Quantity"]).issubset(df_reg_fe.columns):
    df_reg_fe["Price_x_Quantity"] = (
        df_reg_fe["Total_Price"] * df_reg_fe["Ticket_Quantity"]
    )

if "Total_Price" in df_reg_fe.columns:
    df_reg_fe["Sqrt_Total_Price"] = np.sqrt(df_reg_fe["Total_Price"])

df_reg_fe = df_reg_fe.drop(columns=["PCA1", "PCA2"], errors="ignore")

st.dataframe(df_reg_fe.head())

# =====================================================
# REGRESI GLOBAL ENSEMBLE
# =====================================================
st.header("🟢 Regresi Ensemble Global")

Xg = df_reg_fe.drop(columns=["Ticket_Quantity"], errors="ignore")
yg = df_reg_fe["Ticket_Quantity"]

Xg = Xg.select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(
    Xg, yg, test_size=0.2, random_state=42
)

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.01)
elastic = ElasticNet(alpha=0.01, l1_ratio=0.5)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
elastic.fit(X_train, y_train)

y_pred = (
    ridge.predict(X_test) +
    lasso.predict(X_test) +
    elastic.predict(X_test)
) / 3

st.dataframe(pd.DataFrame([{
    "MSE": mean_squared_error(y_test, y_pred),
    "R2_Score": r2_score(y_test, y_pred)
}]))

# =====================================================
# REGRESI ENSEMBLE PER CLUSTER
# =====================================================
st.header("🟢 Regresi Ensemble per Cluster")

results = []

for c in sorted(df_reg_fe["Cluster"].unique()):
    data_c = df_reg_fe[df_reg_fe["Cluster"] == c]

    if len(data_c) < 10:
        continue

    X = data_c.drop(columns=["Ticket_Quantity"], errors="ignore")
    y = data_c["Ticket_Quantity"]

    X = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)
    elastic.fit(X_train, y_train)

    y_pred = (
        ridge.predict(X_test) +
        lasso.predict(X_test) +
        elastic.predict(X_test)
    ) / 3

    results.append({
        "Cluster": c,
        "Jumlah_Data": len(data_c),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2_Score": r2_score(y_test, y_pred)
    })

st.dataframe(pd.DataFrame(results))
