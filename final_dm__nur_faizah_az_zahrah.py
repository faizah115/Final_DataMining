# =====================================================
# FINAL DATA MINING - NUR FAIZAH AZ ZAHRAH
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="UAS Data Mining", layout="wide")
st.title("Analisis Clustering (GMM) & Regresi Ensemble")

# =====================================================
# LOAD DATA
# =====================================================
uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if uploaded_file is None:
    st.info("Silakan upload file CSV untuk memulai analisis.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Data Awal")
st.dataframe(df.head())

# =====================================================
# 1. DATA CLEANING
# =====================================================
st.subheader("Data Cleaning")

st.write("Ukuran Data Awal:", df.shape)
st.write("Missing Value per Kolom:")
st.write(df.isnull().sum())

df = df.drop_duplicates()

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

st.write("Ukuran Data Setelah Cleaning:", df.shape)

# =====================================================
# 2. ENCODING DATA KATEGORIK
# =====================================================
st.subheader("Encoding Data Kategorik")

df_encoded = df.copy()
encoder = LabelEncoder()

for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col] = encoder.fit_transform(df_encoded[col])

st.dataframe(df_encoded.head())

st.download_button(
    "Download Data Encoding (CSV)",
    data=df_encoded.to_csv(index=False),
    file_name="data_encoded.csv",
    mime="text/csv"
)

# =====================================================
# 3. FEATURE ENGINEERING (DATE)
# =====================================================
if "Date" in df_encoded.columns:
    df_encoded["Month"] = df_encoded["Date"].dt.month
    df_encoded["Day"] = df_encoded["Date"].dt.day
    df_encoded["DayOfWeek"] = df_encoded["Date"].dt.dayofweek

# =====================================================
# 4. SCALING DATA
# =====================================================
numeric_cols = df_encoded.select_dtypes(include=np.number)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_cols)

# =====================================================
# A. ANALISIS CLUSTERING (GMM)
# =====================================================
st.header("A. Analisis Clustering (Gaussian Mixture Model)")

# --- GMM
gmm = GaussianMixture(n_components=3, random_state=42)
cluster_labels = gmm.fit_predict(X_scaled)
df_encoded["Cluster"] = cluster_labels

# --- Evaluasi Clustering
sil_score = silhouette_score(X_scaled, cluster_labels)
st.metric("Silhouette Score", round(sil_score, 4))

# --- PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_encoded["PCA1"] = X_pca[:, 0]
df_encoded["PCA2"] = X_pca[:, 1]

fig1, ax1 = plt.subplots()
sns.scatterplot(
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette="Set2",
    data=df_encoded,
    ax=ax1
)
ax1.set_title("PCA Visualization - GMM Clustering")
st.pyplot(fig1)

buf1 = io.BytesIO()
fig1.savefig(buf1, format="png", bbox_inches="tight")
buf1.seek(0)

st.download_button(
    "Download PCA Clustering (PNG)",
    data=buf1,
    file_name="pca_gmm_clustering.png",
    mime="image/png"
)

# =====================================================
# B. REGRESI + ENSEMBLE (SUPERVISED)
# =====================================================
st.header("B. Analisis Regresi + Ensemble")

target = "Ticket_Quantity"

X = df_encoded.drop(
    columns=[target, "Cluster", "PCA1", "PCA2"],
    errors="ignore"
)
y = df_encoded[target]

X = X.select_dtypes(include=np.number)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Model Ensemble
ridge = Ridge()
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

# --- Evaluasi Regresi
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

st.metric("MSE", round(mse, 4))
st.metric("RMSE", round(rmse, 4))

# --- PCA Visualization Regresi
pca_reg = PCA(n_components=2)
X_reg_pca = pca_reg.fit_transform(X_scaled)

fig2, ax2 = plt.subplots()
ax2.scatter(X_reg_pca[:, 0], X_reg_pca[:, 1], alpha=0.6)
ax2.set_title("PCA Visualization - Regression Data")
ax2.set_xlabel("PCA1")
ax2.set_ylabel("PCA2")
st.pyplot(fig2)

buf2 = io.BytesIO()
fig2.savefig(buf2, format="png", bbox_inches="tight")
buf2.seek(0)

st.download_button(
    "Download PCA Regression (PNG)",
    data=buf2,
    file_name="pca_regression.png",
    mime="image/png"
)

# =====================================================
# FINAL DOWNLOAD
# =====================================================
st.subheader("Download Dataset Akhir")

st.download_button(
    "Download Dataset Lengkap (CSV)",
    data=df_encoded.to_csv(index=False),
    file_name="hasil_akhir_clustering_regresi.csv",
    mime="text/csv"
)
