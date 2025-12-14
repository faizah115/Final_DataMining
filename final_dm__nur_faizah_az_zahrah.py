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

else:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Awal")
    st.dataframe(df.head())


# ===============================
# DATA CLEANING
# ===============================
st.header("🧹 Data Cleaning")

# Ukuran data awal
st.write("Ukuran data awal (baris, kolom):", df.shape)

# -------------------------------
# Missing Value
# -------------------------------
st.subheader("Missing Value per Kolom")
missing_df = df.isnull().sum().reset_index()
missing_df.columns = ["Kolom", "Jumlah Missing"]
st.dataframe(missing_df)

# -------------------------------
# Data Duplikat
# -------------------------------
jumlah_duplikat = df.duplicated().sum()
st.write("Jumlah data duplikat sebelum dihapus:", jumlah_duplikat)

# Hapus duplikat
df = df.drop_duplicates()

st.write("Jumlah data duplikat setelah dihapus:", df.duplicated().sum())

# -------------------------------
# Konversi Date
# -------------------------------
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    st.subheader("Contoh Kolom Date Setelah Konversi")
    st.dataframe(df[["Date"]].head())

# Ukuran data akhir
st.write("Ukuran data setelah cleaning:", df.shape)


# ===============================
# ENCODING
# ===============================
df_encoded = df.copy()
encoder = LabelEncoder()
for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col] = encoder.fit_transform(df_encoded[col])


st.subheader("📁 Data Setelah Encoding")

# Tampilkan 5 baris saja
st.write("Contoh 5 baris data hasil encoding:")
st.dataframe(df_encoded.head())

# Simpan ke CSV
csv_encoded = df_encoded.to_csv(index=False).encode("utf-8")

# Tombol download
st.download_button(
    label="Download Data Encoding (CSV)",
    data=csv_encoded,
    file_name="data_encoded.csv",
    mime="text/csv"
)



# ===============================
# FEATURE ENGINEERING (DATE)
# ===============================

# Pastikan kolom Date ada
if "Date" in df_encoded.columns:
    df_encoded["Month"] = df_encoded["Date"].dt.month
    df_encoded["Day"] = df_encoded["Date"].dt.day
    df_encoded["DayOfWeek"] = df_encoded["Date"].dt.dayofweek

# Fitur khusus untuk clustering
fitur_clustering = [
    "Ticket_Quantity",   # volume pembelian
    "Total_Price",       # nilai transaksi
    "Month",             # pola musiman
    "DayOfWeek"          # pola hari
]

# Pastikan fitur benar-benar ada
fitur_clustering = [c for c in fitur_clustering if c in df_encoded.columns]

st.write("Fitur clustering yang digunakan:", fitur_clustering)

X_cluster = df_encoded[fitur_clustering]

# Scaling (WAJIB untuk clustering)
scaler = StandardScaler()
df_encoded["Cluster"] = cluster_model.fit_predict(X_cluster_scaled)



# =====================================================
# ANALISIS 1: CLUSTERING SAJA (AGGLOMERATIVE)
# =====================================================
st.header("🔵 Analisis Clustering (Agglomerative)")

cluster_model = AgglomerativeClustering(n_clusters=3)
df_encoded["Cluster"] = cluster_model.fit_predict(X_scaled)

st.write("Distribusi Cluster:")
st.write(df_encoded["Cluster"].value_counts())

# PCA CLUSTERING
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


st.subheader("Evaluasi Clustering (Silhouette Score)")

silhouette_vals = silhouette_samples(X_scaled, df_encoded["Cluster"])
df_encoded["Silhouette"] = silhouette_vals

cluster_eval = (
    df_encoded
    .groupby("Cluster")
    .agg(
        Jumlah_Data=("Cluster", "count"),
        Silhouette_Score=("Silhouette", "mean")
    )
    .reset_index()
)

st.dataframe(cluster_eval)

st.subheader("Evaluasi Clustering Keseluruhan")

overall_silhouette = silhouette_score(X_scaled, df_encoded["Cluster"])

overall_eval_df = pd.DataFrame({
    "Jumlah_Cluster": [df_encoded["Cluster"].nunique()],
    "Jumlah_Data": [len(df_encoded)],
    "Silhouette_Score_Keseluruhan": [overall_silhouette]
})

st.dataframe(overall_eval_df)


# =====================================================
# ANALISIS 2: REGRESI + ENSEMBLE METHODE
# =====================================================

# =====================================================
# A. REGRESI GLOBAL (KESELURUHAN DATA)
# =====================================================
st.subheader("🟢 Hasil Ensemble Regresi (Keseluruhan Data)")

if df_encoded.shape[0] > 10:

    Xg = df_encoded.drop(
        columns=["Ticket_Quantity", "Cluster", "PCA1", "PCA2"],
        errors="ignore"
    )
    yg = df_encoded["Ticket_Quantity"]

    Xg = Xg.select_dtypes(include=[np.number])

    st.write("Jumlah fitur regresi global:", Xg.shape[1])
    st.write("Jumlah data regresi global:", Xg.shape[0])

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
        ridge.predict(X_test)
        + lasso.predict(X_test)
        + elastic.predict(X_test)
    ) / 3

    global_df = pd.DataFrame([{
        "MSE": mean_squared_error(y_test, y_pred),
        "R2_Score": r2_score(y_test, y_pred),
        "Jumlah_Data": len(df_encoded)
    }])

    st.dataframe(global_df)

else:
    st.warning("Data terlalu sedikit untuk regresi global")


# =====================================================
# B. REGRESI ENSEMBLE PER CLUSTER
# =====================================================
st.subheader("🟢 Hasil Evaluasi Regresi Ensemble per Cluster")

results = []

for c in sorted(df_encoded["Cluster"].unique()):
    data_c = df_encoded[df_encoded["Cluster"] == c]

    st.write(f"Cluster {c} | Jumlah data:", len(data_c))

    if len(data_c) < 10:
        st.warning(f"Cluster {c} dilewati (data terlalu sedikit)")
        continue

    X = data_c.drop(
        columns=["Ticket_Quantity", "Cluster", "PCA1", "PCA2"],
        errors="ignore"
    )
    y = data_c["Ticket_Quantity"]

    X = X.select_dtypes(include=[np.number])

    if X.shape[1] == 0:
        st.warning(f"Cluster {c} dilewati (fitur kosong)")
        continue


# ===============================
# FEATURE ENGINEERING - REGRESI
# ===============================

# Target
target_column = "Ticket_Quantity"

# Fitur regresi (lebih lengkap dari clustering)
fitur_regresi = [
    "Total_Price",
    "Month",
    "Day",
    "DayOfWeek"
]

# Tambahkan fitur numerik lain jika ada
fitur_regresi += list(
    df_encoded.select_dtypes(include=[np.number]).columns
)

# Hapus duplikat & target
fitur_regresi = list(set(fitur_regresi))
fitur_regresi = [c for c in fitur_regresi if c in df_encoded.columns and c != target_column]

st.write("Fitur regresi yang digunakan:", fitur_regresi)

X_regresi = df_encoded[fitur_regresi]
y_regresi = df_encoded[target_column]

# Scaling (WAJIB untuk Ridge/Lasso/ElasticNet)
scaler_reg = StandardScaler()
X_regresi_scaled = scaler_reg.fit_transform(X_regresi)


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

results.append({
        "Cluster": c,
        "Jumlah_Data": len(data_c),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2_Score": r2_score(y_test, y_pred)
    })

# TAMPILKAN HASIL PER CLUSTER
if len(results) > 0:
    result_df = pd.DataFrame(results)
    st.dataframe(result_df)
else:
    st.error("Tidak ada hasil regresi per cluster yang berhasil dihitung")


# PCA REGRESI (VISUALISASI SAJA)
fig_r, ax_r = plt.subplots()
sns.scatterplot(
    data=df_encoded,
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette="Set1",
    ax=ax_r
)
ax_r.set_title("PCA untuk Analisis Regresi Ensemble")
st.pyplot(fig_r)

buf_r = io.BytesIO()
fig_r.savefig(buf_r, format="png", bbox_inches="tight")
buf_r.seek(0)
st.download_button(
    "Download Visualisasi PCA Regresi",
    buf_r,
    "pca_regresi_ensemble.png",
    "image/png"
)

