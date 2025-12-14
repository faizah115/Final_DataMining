# IMPORT LIBRARY
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


st.title("Analisis Clustering dan Regresi Ensemble")

# LOAD DATA
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

# DATA CLEANING
st.header("Data Cleaning")

# Ukuran data awal
st.write("Ukuran data awal (baris, kolom):", df.shape)

# Missing Value
st.subheader("Missing Value per Kolom")
missing_df = df.isnull().sum().reset_index()
missing_df.columns = ["Kolom", "Jumlah Missing"]
st.dataframe(missing_df)

# Data Duplikat
jumlah_duplikat = df.duplicated().sum()
st.write("Jumlah data duplikat sebelum dihapus:", jumlah_duplikat)

# Hapus duplikat
df = df.drop_duplicates()

st.write("Jumlah data duplikat setelah dihapus:", df.duplicated().sum())

# Konversi Date
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    st.subheader("Contoh Kolom Date Setelah Konversi")
    st.dataframe(df[["Date"]].head())

st.write("Ukuran data setelah cleaning:", df.shape)

"""
Pada tahap Data Cleaning tersebut, beberapa langkah pembersihan dan pengecekan kualitas data dilakukan untuk memastikan dataset siap dianalisis. Pertama, ditampilkan ukuran data awal untuk mengetahui jumlah baris dan kolom sebelum proses pembersihan. Selanjutnya, dilakukan pemeriksaan missing value pada setiap kolom menggunakan df.isnull().sum() untuk memastikan tidak ada nilai kosong yang dapat memengaruhi hasil analisis. Setelah itu, data duplikat dihapus menggunakan drop_duplicates() agar tidak terjadi pengulangan data yang dapat menimbulkan bias. Kemudian, jika terdapat kolom Date, nilainya dikonversi ke format datetime agar dapat digunakan dengan benar pada proses analisis lanjutan, seperti feature engineering berbasis waktu. Terakhir, ditampilkan kembali ukuran data setelah cleaning untuk memastikan bahwa proses pembersihan telah berjalan dengan baik dan untuk melihat apakah terjadi perubahan jumlah data.
"""


# FEATURE ENGINEERING (TIME-BASED)
if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):

    # Feature engineering waktu
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    st.subheader("Visualisasi Fitur Waktu")

    col1, col2, col3 = st.columns(3)

    # Distribusi Transaksi per Bulan
    with col1:
        fig_m, ax_m = plt.subplots()
        sns.countplot(x="Month", data=df, ax=ax_m)
        ax_m.set_title("Distribusi Transaksi per Bulan")
        ax_m.set_xlabel("Bulan")
        ax_m.set_ylabel("Jumlah Data")
        st.pyplot(fig_m)

    # Distribusi Transaksi per Hari
    with col2:
        fig_d, ax_d = plt.subplots()
        sns.countplot(x="Day", data=df, ax=ax_d)
        ax_d.set_title("Distribusi Transaksi per Hari")
        ax_d.set_xlabel("Hari")
        ax_d.set_ylabel("Jumlah Data")
        st.pyplot(fig_d)

    # Distribusi Transaksi per Hari dalam Minggu
    with col3:
        fig_w, ax_w = plt.subplots()
        sns.countplot(x="DayOfWeek", data=df, ax=ax_w)
        ax_w.set_title("Distribusi Transaksi per Hari dalam Minggu")
        ax_w.set_xlabel("Hari (0=Senin)")
        ax_w.set_ylabel("Jumlah Data")
        st.pyplot(fig_w)



"""Distribusi Transaksi per Bulan
Grafik distribusi transaksi per bulan menunjukkan adanya perbedaan jumlah transaksi pada tiap bulan, di mana bulan pertama memiliki jumlah transaksi paling tinggi dibandingkan bulan lainnya. Hal ini mengindikasikan adanya pola musiman dalam data, sehingga fitur bulan relevan digunakan untuk menangkap variasi perilaku transaksi berdasarkan waktu dan dapat mendukung proses clustering maupun regresi.

Distribusi transaksi berdasarkan hari dalam bulan memperlihatkan fluktuasi jumlah transaksi yang cukup bervariasi pada setiap tanggal. Tidak semua hari memiliki intensitas transaksi yang sama, yang menunjukkan bahwa aktivitas transaksi tidak bersifat acak dan berpotensi dipengaruhi oleh faktor waktu tertentu, sehingga fitur hari dapat memberikan informasi tambahan terutama dalam pemodelan regresi.

Distribusi Transaksi per Hari dalam Minggu
Grafik distribusi transaksi per hari dalam minggu menunjukkan perbedaan jumlah transaksi antar hari, dengan beberapa hari memiliki intensitas transaksi lebih tinggi dibandingkan hari lainnya. Pola ini mengindikasikan bahwa perilaku transaksi dipengaruhi oleh hari kerja tertentu, sehingga fitur hari dalam minggu menjadi fitur temporal yang penting dan relevan untuk analisis clustering dan regresi.
"""



# ENCODING
df_encoded = df.copy()
encoder = LabelEncoder()
for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col] = encoder.fit_transform(df_encoded[col])


st.subheader("Data Setelah Encoding")
st.write("Contoh 5 baris data hasil encoding:")
st.dataframe(df_encoded.head())
csv_encoded = df_encoded.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Data Encoding (CSV)",
    data=csv_encoded,
    file_name="data_encoded.csv",
    mime="text/csv"
)



"""
Output dari proses ini adalah dataset dengan struktur dan jumlah data yang sama seperti sebelumnya (500 baris dan 9 kolom), namun seluruh kolom kategorik seperti City, Gender, Airline, dan Payment_Method telah diubah menjadi nilai numerik. Perubahan ini tidak memengaruhi jumlah data, hanya mengubah representasi nilai agar sesuai dengan kebutuhan algoritma analisis. File data_encoded.csv yang dihasilkan menjadi bukti bahwa data kategorik telah berhasil dipreproses dan siap digunakan pada tahap pemodelan selanjutnya.
"""


# FEATURE ENGINEERING (DATE)
df_encoded["Month"] = df_encoded["Date"].dt.month
df_encoded["Day"] = df_encoded["Date"].dt.day
df_encoded["DayOfWeek"] = df_encoded["Date"].dt.dayofweek

fitur_clustering = [
    "Ticket_Quantity",
    "Total_Price",
    "Month",
    "DayOfWeek"
]

fitur_clustering = [c for c in fitur_clustering if c in df_encoded.columns]

st.write("Fitur clustering yang digunakan:", fitur_clustering)

X = df_encoded[fitur_clustering]
X_scaled = StandardScaler().fit_transform(X)

"""
Makna Output ['Ticket_Quantity', 'Month', 'DayOfWeek']
Output tersebut menunjukkan bahwa fitur yang berhasil dipilih dan digunakan dalam proses clustering (Agglomerative) terdiri dari tiga variabel, yaitu Ticket_Quantity, Month, dan DayOfWeek. Artinya, hanya ketiga fitur ini yang tersedia, valid, dan memenuhi kriteria sebagai fitur numerik setelah proses feature engineering dan encoding. Dengan demikian, pembentukan cluster didasarkan pada jumlah tiket yang dibeli serta pola waktu transaksi berdasarkan bulan dan hari dalam minggu, sehingga cluster yang dihasilkan merepresentasikan kelompok data dengan karakteristik perilaku transaksi dan pola temporal yang serupa.
"""


# ANALISIS 1: CLUSTERING GMM (AGGLOMERATIVE)
st.header("Analisis Clustering (Agglomerative)")

cluster_model = AgglomerativeClustering(n_clusters=3)
df_encoded["Cluster"] = cluster_model.fit_predict(X_scaled)

st.write("Distribusi Cluster:")
st.write(df_encoded["Cluster"].value_counts())


"""
Distribusi Jumlah Data per Cluster
Tabel distribusi cluster menunjukkan bahwa hasil clustering Agglomerative membagi data ke dalam tiga kelompok, yaitu Cluster 0 sebanyak 238 data, Cluster 1 sebanyak 151 data, dan Cluster 2 sebanyak 111 data. Perbedaan jumlah anggota pada setiap cluster menandakan adanya variasi karakteristik data yang berhasil ditangkap oleh model, di mana Cluster 0 merupakan kelompok dominan dengan jumlah data terbesar, sedangkan Cluster 2 memiliki jumlah data paling sedikit, sehingga masing-masing cluster merepresentasikan segmen data dengan pola perilaku transaksi yang berbeda
"""

# PCA CLUSTERING
pca_cluster = PCA(n_components=2)
X_pca_cluster = pca_cluster.fit_transform(X_scaled)
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


"""
Visualisasi PCA Clustering (Agglomerative)
Visualisasi PCA menunjukkan hasil clustering Agglomerative yang diproyeksikan ke dalam dua komponen utama, yaitu PCA1 dan PCA2, sehingga pola pemisahan antar cluster dapat diamati secara visual. Terlihat bahwa data terbagi ke dalam tiga kelompok dengan warna yang berbeda, di mana masing-masing cluster memiliki kecenderungan posisi tertentu meskipun masih terdapat beberapa titik yang saling berdekatan. Hal ini menunjukkan bahwa model clustering mampu mengelompokkan data berdasarkan kesamaan karakteristik fitur, namun antar cluster masih memiliki hubungan yang relatif dekat, yang wajar terjadi pada data perilaku transaksi yang memiliki variasi bertahap.
"""


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
"""
Evaluasi Clustering Menggunakan Silhouette Score
Hasil evaluasi clustering menggunakan Silhouette Score menunjukkan bahwa Cluster 2 memiliki nilai silhouette tertinggi sebesar 0,3784, yang mengindikasikan bahwa anggota pada cluster ini memiliki tingkat kekompakan dan pemisahan yang paling baik dibandingkan cluster lainnya. Cluster 1 memiliki nilai silhouette sebesar 0,2531 yang menunjukkan kualitas pemisahan sedang, sedangkan Cluster 0 memiliki nilai silhouette terendah sebesar 0,1628, yang menandakan bahwa sebagian data pada cluster tersebut masih cukup dekat dengan cluster lain. Secara keseluruhan, hasil ini menunjukkan bahwa model Agglomerative Clustering mampu membentuk kelompok data, namun kualitas pemisahan antar cluster masih tergolong moderat.
"""

st.dataframe(cluster_eval)

st.subheader("Evaluasi Clustering Keseluruhan")

overall_silhouette = silhouette_score(X_scaled, df_encoded["Cluster"])

overall_eval_df = pd.DataFrame({
    "Jumlah_Cluster": [df_encoded["Cluster"].nunique()],
    "Jumlah_Data": [len(df_encoded)],
    "Silhouette_Score_Keseluruhan": [overall_silhouette]
})

st.dataframe(overall_eval_df)

"""
Evaluasi Clustering Keseluruhan
Hasil evaluasi clustering secara keseluruhan menunjukkan bahwa model Agglomerative Clustering dengan jumlah 3 cluster yang diterapkan pada 500 data menghasilkan nilai Silhouette Score sebesar 0,2379. Nilai ini mengindikasikan bahwa kualitas pemisahan antar cluster berada pada tingkat sedang, di mana struktur cluster sudah terbentuk namun masih terdapat beberapa tumpang tindih antar kelompok. Secara umum, hasil ini menunjukkan bahwa model mampu menangkap pola dasar dalam data, meskipun masih terdapat ruang untuk peningkatan kualitas clustering melalui penyesuaian fitur atau jumlah cluster.
"""




# A. REGRESI GLOBAL (KESELURUHAN DATA)
st.subheader(" Hasil Ensemble Regresi (Keseluruhan Data)")

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

"""
Hasil Evaluasi Regresi Ensemble (Keseluruhan Data)
Hasil evaluasi regresi ensemble pada keseluruhan data menunjukkan nilai Mean Squared Error (MSE) sebesar 0,2194 dan nilai R² sebesar 0,889 dengan jumlah data sebanyak 500. Nilai R² yang tinggi mengindikasikan bahwa model regresi ensemble mampu menjelaskan sekitar 88,9% variasi pada variabel target, sehingga memiliki kemampuan prediksi yang sangat baik. Sementara itu, nilai MSE yang relatif kecil menunjukkan bahwa kesalahan prediksi yang dihasilkan model tergolong rendah, sehingga model regresi ensemble dapat dikatakan efektif dalam memodelkan hubungan antara fitur dan jumlah tiket.
"""

# B. REGRESI ENSEMBLE PER CLUSTER
st.subheader("Hasil Evaluasi Regresi Ensemble per Cluster")

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


"""
Hasil Evaluasi Regresi Ensemble per Cluster
Hasil evaluasi regresi ensemble per cluster menunjukkan bahwa setiap cluster memiliki tingkat akurasi prediksi yang baik, ditunjukkan oleh nilai R² yang tinggi pada seluruh cluster. Cluster 1 memiliki performa terbaik dengan nilai R² sebesar 0,9083 dan MSE terendah sebesar 0,0561, yang menandakan model sangat mampu memprediksi jumlah tiket pada kelompok data tersebut. Cluster 0 juga menunjukkan performa yang baik dengan nilai R² sebesar 0,8878 dan MSE sebesar 0,056, sementara Cluster 2 memiliki nilai R² sedikit lebih rendah yaitu 0,8565 dan MSE sebesar 0,0868, yang mengindikasikan bahwa variasi data pada cluster ini lebih kompleks namun tetap dapat dimodelkan dengan cukup baik.
"""


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

"""
Visualisasi PCA untuk Analisis Regresi Ensemble
Visualisasi PCA pada analisis regresi ensemble menampilkan sebaran data yang telah diproyeksikan ke dalam dua komponen utama, yaitu PCA1 dan PCA2, dengan pewarnaan berdasarkan cluster hasil Agglomerative Clustering. Grafik ini menunjukkan bahwa masing-masing cluster memiliki kecenderungan wilayah sebaran tertentu meskipun masih terdapat tumpang tindih antar cluster, yang mencerminkan adanya hubungan yang saling berdekatan antar kelompok data. Visualisasi ini digunakan sebagai alat bantu untuk memahami struktur data sebelum dan sesudah pemodelan regresi, serta mendukung interpretasi bahwa regresi ensemble dilakukan pada data yang telah tersegmentasi dengan baik berdasarkan karakteristik cluster.
"""