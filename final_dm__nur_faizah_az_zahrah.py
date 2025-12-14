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
uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

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

st.write("Ukuran data awal:", df.shape)

missing_df = df.isnull().sum().reset_index()
missing_df.columns = ["Kolom", "Jumlah Missing"]
st.dataframe(missing_df)

jumlah_duplikat = df.duplicated().sum()
st.write("Duplikat sebelum:", jumlah_duplikat)
df = df.drop_duplicates()
st.write("Duplikat sesudah:", df.duplicated().sum())

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    st.dataframe(df[["Date"]].head())

# ===============================
# ENCODING
# ===============================
df_encoded = df.copy()
encoder = LabelEncoder()
for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col] = encoder.fit_transform(df_encoded[col])

st.subheader("📁 Data Encoding (5 Baris)")
st.dataframe(df_encoded.head())

st.download_button(
    "Download Data Encoding",
    df_encoded.to_csv(index=False),
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
# FEATURE ENGINEERING - CLUSTERING
# ===============================
fitur_clustering = [
    "Ticket_Quantity",
    "Total_Price",
    "Month",
    "DayOfWeek"
]
fitur_clustering = [c for c in fitur_clustering if c in df_encoded.columns]

st.write("Fitur Clustering:", fitur_clustering)

X_cluster = df_encoded[fitur_clustering]
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# ===============================
# ANALISIS 1: CLUSTERING
# ===============================
st.header("🔵 Analisis Clustering (Agglomerative)")

cluster_model = AgglomerativeClustering(n_clusters=3)
df_encoded["Cluster"] = cluster_model.fit_predict(X_cluster_scaled)

st.write(df_encoded["Cluster"].value_counts())

# PCA CLUSTERING
pca_cluster = PCA(n_components=2)
X_pca_cluster = pca_cluster.fit_transform(X_cluster_scaled)
df_encoded["PCA1"] = X_pca_cluster[:, 0]
df_encoded["PCA2"] = X_pca_cluster[:, 1]

fig_c, ax_c = plt.subplots()
sns.scatterplot(data=df_encoded, x="PCA1", y="PCA2", hue="Cluster", ax=ax_c)
st.pyplot(fig_c)

# Silhouette
st.subheader("Evaluasi Clustering")

df_encoded["Silhouette"] = silhouette_samples(X_cluster_scaled, df_encoded["Cluster"])

cluster_eval = df_encoded.groupby("Cluster").agg(
    Jumlah_Data=("Cluster", "count"),
    Silhouette_Score=("Silhouette", "mean")
).reset_index()

st.dataframe(cluster_eval)

overall_eval = pd.DataFrame({
    "Jumlah_Cluster": [df_encoded["Cluster"].nunique()],
    "Jumlah_Data": [len(df_encoded)],
    "Silhouette_Keseluruhan": [
        silhouette_score(X_cluster_scaled, df_encoded["Cluster"])
    ]
})

st.dataframe(overall_eval)

# ===============================
# FEATURE ENGINEERING - REGRESI
# ===============================
st.header("🟢 Analisis Regresi + Ensemble")

target_column = "Ticket_Quantity"

fitur_regresi = [
    "Total_Price",
    "Month",
    "Day",
    "DayOfWeek"
]

fitur_regresi += list(df_encoded.select_dtypes(include=[np.number]).columns)
fitur_regresi = list(set(fitur_regresi))
fitur_regresi = [c for c in fitur_regresi if c != target_column]

st.write("Fitur Regresi:", fitur_regresi)

X_reg = df_encoded[fitur_regresi]
y_reg = df_encoded[target_column]

scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

# ===============================
# REGRESI GLOBAL
# ===============================
st.subheader("Regresi Ensemble Global")

X_train, X_test, y_train, y_test = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42
)

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.01)
elastic = ElasticNet(alpha=0.01, l1_ratio=0.5)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
elastic.fit(X_train, y_train)

y_pred = (ridge.predict(X_test) + lasso.predict(X_test) + elastic.predict(X_test)) / 3

global_df = pd.DataFrame([{
    "MSE": mean_squared_error(y_test, y_pred),
    "R2_Score": r2_score(y_test, y_pred),
    "Jumlah_Data": len(df_encoded)
}])

st.dataframe(global_df)

# ===============================
# REGRESI PER CLUSTER
# ===============================
st.subheader("Regresi Ensemble per Cluster")

results = []

for c in sorted(df_encoded["Cluster"].unique()):
    data_c = df_encoded[df_encoded["Cluster"] == c]

    if len(data_c) < 10:
        continue

    Xc = data_c[fitur_regresi]
    yc = data_c[target_column]

    Xc_scaled = scaler_reg.fit_transform(Xc)

    X_train, X_test, y_train, y_test = train_test_split(
        Xc_scaled, yc, test_size=0.2, random_state=42
    )

    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)
    elastic.fit(X_train, y_train)

    y_pred = (ridge.predict(X_test) + lasso.predict(X_test) + elastic.predict(X_test)) / 3

    results.append({
        "Cluster": c,
        "Jumlah_Data": len(data_c),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2_Score": r2_score(y_test, y_pred)
    })

st.dataframe(pd.DataFrame(results))
