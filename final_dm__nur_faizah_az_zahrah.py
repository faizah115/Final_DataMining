# =========================================
# IMPORT LIBRARY
# =========================================
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

# =========================================
# CONFIG STREAMLIT
# =========================================
st.set_page_config(page_title="UAS Data Mining", layout="wide")
st.title("Clustering & Ensemble Regression")
st.write("Agglomerative Clustering + Ensemble Regression")

# =========================================
# LOAD DATA
# =========================================
uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if uploaded_file is None:
    st.info("Silakan upload dataset CSV")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("Data Awal")
st.dataframe(df.head())

# =========================================
# 1. DATA CLEANING
# =========================================
st.subheader("🧹 Data Cleaning")

st.write("Ukuran data awal:", df.shape)
st.write("Missing value per kolom:")
st.write(df.isnull().sum())

st.write("Jumlah duplikat:", df.duplicated().sum())
df = df.drop_duplicates()

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

st.write("Ukuran data setelah cleaning:", df.shape)

# =========================================
# 2. ENCODING DATA KATEGORIK
# =========================================
st.subheader("🔤 Encoding Data Kategorik")

df_encoded = df.copy()
encoder = LabelEncoder()

for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col] = encoder.fit_transform(df_encoded[col])

st.dataframe(df_encoded.head())

st.download_button(
    "Download Data Encoding",
    data=df_encoded.to_csv(index=False),
    file_name="data_encoded.csv",
    mime="text/csv"
)

# =========================================
# 3. SCALING DATA
# =========================================
st.subheader("⚖️ Scaling Data")

datetime_cols = df_encoded.select_dtypes(include=["datetime64[ns]"]).columns
df_numeric = df_encoded.drop(columns=datetime_cols)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# =========================================
# 4. FEATURE ENGINEERING (UNTUK CLUSTERING)
# =========================================
st.subheader("🧩 Feature Engineering (Clustering)")

cluster_features = df_numeric.copy()
st.write("Fitur yang digunakan untuk clustering:")
st.write(cluster_features.columns.tolist())

# =========================================
# 5. CLUSTERING (AGGLOMERATIVE)
# =========================================
st.subheader("🔹 Agglomerative Clustering")

n_cluster = st.slider("Pilih jumlah cluster", 2, 5, 3)

cluster_model = AgglomerativeClustering(n_clusters=n_cluster)
df["Cluster"] = cluster_model.fit_predict(X_scaled)

st.write("Distribusi Cluster:")
st.write(df["Cluster"].value_counts())

# =========================================
# PCA UNTUK VISUALISASI CLUSTERING
# =========================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

st.subheader("Visualisasi PCA Clustering")

fig_pca, ax_pca = plt.subplots()
sns.scatterplot(
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    data=df,
    palette="Set2",
    ax=ax_pca
)
ax_pca.set_title("PCA Agglomerative Clustering")
st.pyplot(fig_pca)

# Download visualisasi PCA
buf = io.BytesIO()
fig_pca.savefig(buf, format="png", bbox_inches="tight")
buf.seek(0)

st.download_button(
    "Download Visualisasi PCA Clustering",
    data=buf,
    file_name="pca_clustering.png",
    mime="image/png"
)

# =========================================
# BOXPLOT CLUSTERING
# =========================================
st.subheader("📦 Boxplot Clustering")

fig_box_c, ax_box_c = plt.subplots()
sns.boxplot(
    x="Cluster",
    y="Ticket_Quantity",
    data=df,
    palette="Set2",
    ax=ax_box_c
)
ax_box_c.set_title("Boxplot Ticket Quantity per Cluster")
st.pyplot(fig_box_c)

# =========================================
# 6. FEATURE ENGINEERING (UNTUK REGRESI)
# =========================================
st.subheader("🧩 Feature Engineering (Regresi)")

if "Date" in df.columns:
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek

# =========================================
# 7. REGRESI + ENSEMBLE
# =========================================
st.subheader("📈 Ensemble Regression")

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

    # BOXPLOT REGRESI
    df_compare = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })

    df_melt = df_compare.melt(
        var_name="Type",
        value_name=target_column
    )

    st.subheader(f"Boxplot Regresi Ensemble - Cluster {cluster}")

    fig_box_r, ax_box_r = plt.subplots()
    sns.boxplot(
        x="Type",
        y=target_column,
        data=df_melt,
        palette="Set3",
        ax=ax_box_r
    )
    ax_box_r.set_title(f"Actual vs Predicted (Cluster {cluster})")
    st.pyplot(fig_box_r)

# =========================================
# HASIL REGRESI
# =========================================
st.subheader("📊 Hasil Ensemble Regression per Cluster")
ensemble_df = pd.DataFrame(ensemble_results)
st.dataframe(ensemble_df)

# =========================================
# DOWNLOAD OUTPUT AKHIR
# =========================================
st.subheader("⬇️ Download Output Akhir")

st.download_button(
    "Download Dataset Final",
    data=df.to_csv(index=False),
    file_name="hasil_clustering_regresi.csv",
    mime="text/csv"
)
