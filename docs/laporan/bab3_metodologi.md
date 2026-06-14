[Kembali](index.md)

# BAB III
# METODOLOGI PENELITIAN

---

## 3.1 Jenis Penelitian dan Dataset

### 3.1.1 Jenis Penelitian

Penelitian ini menggunakan pendekatan **kuantitatif eksperimental** dengan paradigma *data-driven*. Secara metodologis, penelitian ini termasuk dalam kategori penelitian terapan (*applied research*) yang bertujuan membangun dan mengevaluasi sistem klasifikasi berbasis Machine Learning untuk memprediksi status kesejahteraan digital remaja. Kerangka kerja yang diadopsi adalah **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*), yang mencakup enam fase berurutan: *Business Understanding*, *Data Understanding*, *Data Preparation*, *Modeling*, *Evaluation*, dan *Deployment*.

Setiap fase CRISP-DM dalam penelitian ini diimplementasikan secara langsung ke dalam tujuh Jupyter Notebook yang terstruktur (NB1–NB7), sehingga seluruh proses bersifat reproducible dan dapat diaudit secara teknis.

### 3.1.2 Sumber dan Deskripsi Dataset

Dataset yang digunakan adalah **Teen Mental Health Dataset** yang diperoleh dari platform Kaggle (Argonnxx, 2026). Dataset ini berisi 1.200 rekaman data remaja berusia 13 hingga 19 tahun yang mencakup tiga kelompok variabel: (1) informasi demografis, (2) perilaku digital, dan (3) indikator kesehatan mental. Dataset bersifat **sintetis**, artinya dibuat secara artifisial dengan aturan deterministik dan bukan berasal dari pengukuran langsung di lapangan. Implikasi dari sifat sintetis ini dibahas lebih lanjut pada BAB IV.

Tidak ditemukan *missing value* maupun data duplikat pada dataset, sehingga seluruh 1.200 sampel dapat digunakan dalam proses analisis. Proses verifikasi ini dilakukan pada NB1 (`01_data_loading.ipynb`) menggunakan `df.isnull().sum()` dan `df.duplicated().sum()`.

### 3.1.3 Atribut Dataset

Dataset terdiri dari 16 atribut awal. Rincian selengkapnya disajikan pada Tabel 3.1. Informasi lebih lengkap mengenai rentang nilai, kategori, dan catatan teknis tiap atribut tersedia pada dokumen [`data_dictionary.md`](../data_dictionary.md).

**Tabel 3.1 Struktur dan Atribut Dataset Teen Mental Health**

| No | Nama Atribut | Jenis Data | Deskripsi |
|---|---|---|---|
| 1 | `age` | Numerik (Integer) | Usia responden dalam tahun (rentang 13–19) |
| 2 | `gender` | Kategorikal Nominal | Jenis kelamin: male, female |
| 3 | `daily_social_media_hours` | Numerik (Float) | Rata-rata jam penggunaan media sosial per hari (1.0–8.0+) |
| 4 | `platform_usage` | Kategorikal Nominal | Platform yang paling sering digunakan (TikTok, Instagram, YouTube, dst.) |
| 5 | `sleep_hours` | Numerik (Float) | Rata-rata jam tidur per malam (4.0–9.0) |
| 6 | `screen_time_before_sleep` | Numerik (Float) | Durasi penggunaan layar sebelum tidur dalam jam (0.5–3.0) |
| 7 | `academic_performance` | Numerik (Float) | Skor performa akademik ekuivalen GPA (2.00–4.00) |
| 8 | `physical_activity` | Numerik (Float) | Durasi aktivitas fisik per hari dalam jam (0–2.0) |
| 9 | `social_interaction_level` | Kategorikal Ordinal | Tingkat interaksi sosial nyata: low, medium, high |
| 10 | `stress_level` | Numerik (Integer) | Tingkat stres skala 1–10, bersifat *self-report* |
| 11 | `anxiety_level` | Numerik (Integer) | Tingkat kecemasan skala 1–10, bersifat *self-report* |
| 12 | `addiction_level` | Numerik (Integer) | Tingkat perilaku adiktif terhadap gadget skala 1–10 |
| 13 | `depression_label` | Numerik (Biner) | Indikator depresi: 0 (tidak), 1 (ya) |
| 14 | `mental_health_risk_score` | Numerik (Integer) | Skor risiko kesehatan mental agregat (3–30) |
| 15 | `sleep_quality` | Kategorikal Ordinal | Kategori kualitas tidur: Poor, Fair, Good |
| 16 | `digital_wellbeing_flag` | Kategorikal Nominal | **Variabel target**: Healthy, Moderate, At Risk |

Distribusi variabel target `digital_wellbeing_flag` tidak seimbang: kelas *Moderate* mendominasi dengan 746 sampel (62.2%), diikuti *Healthy* sebanyak 364 sampel (30.3%), dan *At Risk* sebanyak 90 sampel (7.5%). Kondisi *imbalanced class* ini ditangani melalui parameter `class_weight='balanced'` pada saat training model (NB5).

---

## 3.2 Sistematika Penyelesaian Masalah

Alur penyelesaian masalah dalam penelitian ini mengikuti kerangka CRISP-DM yang diimplementasikan secara bertahap melalui tujuh notebook. Diagram alir lengkap tersedia pada dokumen [`flowchart.md`](../flowchart.md).

---

*[TEMPAT DIAGRAM FLOWCHART SISTEMATIKA PENELITIAN]*

*(Ganti area ini dengan gambar diagram alir proses penelitian dari `docs/flowchart.md`)*

---

**Gambar 3.1** Flowchart Sistematika Penyelesaian Masalah

---

Berikut penjelasan setiap tahap dalam alur kerja tersebut.

**Tahap 1 — Data Loading dan Understanding (NB1)**

Dataset dimuat dari file `Teen_Mental_Health.csv` menggunakan Pandas. Pada tahap ini dilakukan inspeksi awal meliputi pengecekan tipe data (`df.info()`), statistik deskriptif (`df.describe()`), jumlah *missing value* (`df.isnull().sum()`), dan deteksi duplikat (`df.duplicated().sum()`). Tahap ini menjadi dasar pengambilan keputusan pada tahap berikutnya.

**Tahap 2 — Data Cleaning dan Feature Engineering (NB2)**

Dua kolom dihapus pada tahap ini. Pertama, `platform_usage` dihapus karena tidak deterministik terhadap target. Kedua, `depression_label` dihapus karena teridentifikasi sebagai potensi *data leakage*, yaitu variabel yang secara langsung merepresentasikan kondisi depresi dan beririsan dengan konstruksi target `digital_wellbeing_flag`.

Selanjutnya, lima fitur baru diturunkan dari fitur yang telah ada sebagaimana disajikan pada Tabel 3.2. Proses encoding juga dilakukan pada tahap ini: Label Encoding untuk `sleep_quality` (bersifat ordinal) dan One-Hot Encoding dengan `drop_first=True` untuk `gender` guna menghindari *dummy variable trap*. Hasil akhir disimpan sebagai `clean_data.csv`.

**Tabel 3.2 Fitur Baru Hasil Feature Engineering (NB2)**

| No | Nama Fitur | Formula | Justifikasi |
|---|---|---|---|
| 1 | `total_screen_exposure` | `daily_social_media_hours + screen_time_before_sleep` | Merepresentasikan total paparan layar harian |
| 2 | `sleep_efficiency` | `sleep_hours / (screen_time_before_sleep + 1)` | Rasio kualitas tidur terhadap gangguan layar malam |
| 3 | `high_social_media_usage` | `1 if daily_social_media_hours > 5 else 0` | Flag biner penanda penggunaan berlebih |
| 4 | `active_lifestyle` | `1 if physical_activity >= 1 else 0` | Flag biner penanda gaya hidup aktif secara fisik |
| 5 | `risk_category` | Kuartil `mental_health_risk_score`: Q1≤13=Low, 14–20=Medium, >20=High | Segmentasi risiko berbasis distribusi aktual data |

**Tahap 3 — Exploratory Data Analysis / EDA (NB3)**

EDA dilakukan untuk mengidentifikasi pola dan distribusi data secara visual. Visualisasi yang dihasilkan mencakup: (1) distribusi kelas target menggunakan *count plot*, (2) hubungan antara durasi penggunaan media sosial dengan status kesejahteraan menggunakan *boxplot*, dan (3) korelasi antar fitur numerik menggunakan *heatmap*. Temuan EDA menjadi basis justifikasi pemilihan fitur dan pendekatan pemodelan.

**Tahap 4 — Preprocessing (NB4)**

Variabel target `digital_wellbeing_flag` diubah menjadi numerik: Healthy=0, Moderate=1, At Risk=2. Kolom `social_interaction_level` yang sebelumnya belum diproses juga di-encode menggunakan Label Encoding. Kemudian, tiga kolom yang berpotensi menyebabkan *data leakage* dihapus dari fitur input: `mental_health_risk_score`, `risk_category`, dan `sleep_quality`. Hasilnya disimpan sebagai `preprocessed_data.csv` dalam kondisi belum di-*scale*, karena proses normalisasi yang benar harus dilakukan setelah pemisahan data (lihat Tahap 5). Jumlah fitur final yang digunakan sebagai input model adalah **16 fitur**.

**Tahap 5 — Modeling (NB5)**

Data dibagi dengan proporsi 80:20 menggunakan `train_test_split` dengan parameter `stratify=y` untuk memastikan distribusi kelas tetap proporsional. Normalisasi fitur dilakukan menggunakan `StandardScaler` yang di-*fit* **hanya pada data latih** (`X_train`), kemudian diterapkan (*transform*) pada data latih dan data uji secara terpisah. Objek *scaler* disimpan sebagai `models/scaler.pkl` untuk digunakan kembali pada tahap evaluasi dan deployment.

Model yang digunakan adalah `RandomForestClassifier` dengan konfigurasi `n_estimators=100`, `random_state=42`, dan `class_weight='balanced'`. Validasi model dilakukan menggunakan `StratifiedKFold` dengan K=5. Model terlatih disimpan sebagai `models/classifier_model.pkl`.

**Tahap 6 — Evaluation (NB6)**

Model dievaluasi menggunakan data uji (*hold-out test set*) yang belum pernah dilihat model selama training. Metrik evaluasi yang digunakan mencakup *Classification Report* (Accuracy, Precision, Recall, F1-Score), *Confusion Matrix*, dan analisis *Feature Importance*. Ringkasan hasil K-Fold CV juga ditampilkan ulang sebagai konfirmasi konsistensi model.

**Tahap 7 — Deployment (NB7)**

Model prediksi diimplementasikan dalam bentuk aplikasi web interaktif menggunakan framework Streamlit melalui `%%writefile app.py`. Fungsi `preprocess_input()` di dalam aplikasi mereplikasi seluruh pipeline feature engineering dari NB2 dan proses scaling dari NB5, memastikan data input dari pengguna diproses dalam format yang identik dengan data training. Aplikasi dapat dijalankan secara lokal dengan perintah `streamlit run app.py`.

---

## 3.3 Kerangka Sampling

### 3.3.1 Pembagian Data Latih dan Uji

Penelitian ini menggunakan teknik **Hold-out Sampling** dengan proporsi **80:20**, yaitu 80% data digunakan sebagai data latih (*training set*) dan 20% sebagai data uji (*test set*). Dari total 1.200 sampel, pembagian ini menghasilkan 960 sampel pada data latih dan 240 sampel pada data uji.

Rasio 80:20 dipilih berdasarkan pertimbangan berikut. Pertama, dengan jumlah sampel sebesar 1.200, porsi 80% sudah cukup memberikan volume data yang memadai bagi model untuk mempelajari pola antar fitur secara representatif. Kedua, porsi uji sebesar 20% (240 sampel) menghasilkan set evaluasi yang cukup besar untuk menghasilkan estimasi performa yang stabil, termasuk untuk kelas minoritas *At Risk* yang hanya memiliki 90 sampel di keseluruhan dataset.

Pembagian data dilakukan menggunakan `train_test_split` dari Scikit-learn dengan parameter `stratify=y`, yang memastikan proporsi ketiga kelas target tetap proporsional pada kedua subset data. Tanpa stratifikasi, sampel kelas *At Risk* yang hanya berjumlah 90 data berisiko tidak terwakili secara proporsional, yang dapat menyebabkan bias pada evaluasi model. Rincian distribusi kelas pada masing-masing subset disajikan pada Tabel 3.3.

**Tabel 3.3 Distribusi Kelas pada Data Latih dan Data Uji**

| Subset | Total Sampel | Healthy (0) | Moderate (1) | At Risk (2) |
|---|---|---|---|---|
| Training Set | 960 | 291 (30.3%) | 597 (62.2%) | 72 (7.5%) |
| Test Set | 240 | 73 (30.4%) | 149 (62.1%) | 18 (7.5%) |
| **Total** | **1.200** | **364 (30.3%)** | **746 (62.2%)** | **90 (7.5%)** |

### 3.3.2 Validasi Silang dengan K-Fold

Selain *hold-out test set*, penelitian ini menerapkan **Stratified K-Fold Cross-Validation** dengan K=5 pada data latih untuk mendapatkan estimasi performa model yang lebih andal dan tidak bergantung pada satu pembagian data tertentu. Dengan K=5, data latih (960 sampel) dibagi menjadi 5 *fold* yang masing-masing berisi 192 sampel. Model dilatih dan divalidasi sebanyak 5 kali, dengan setiap iterasi menggunakan 4 *fold* (768 sampel) sebagai data training dan 1 *fold* (192 sampel) sebagai data validasi. Hasil akhir yang dilaporkan adalah rata-rata dan standar deviasi dari kelima iterasi tersebut.

Penggunaan varian *Stratified* K-Fold dipilih untuk menjaga proporsi kelas tetap seimbang di setiap *fold*, mengingat adanya ketidakseimbangan kelas pada dataset (khususnya kelas *At Risk*). Implementasi menggunakan `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` dari Scikit-learn, dikombinasikan dengan `cross_validate` untuk menghasilkan metrik Accuracy, F1-Score, Precision, dan Recall secara bersamaan.

Skema validasi dua lapis ini (K-Fold CV + hold-out test set) memastikan bahwa estimasi performa yang dilaporkan tidak mengalami *optimistic bias* akibat evaluasi pada data yang sama dengan yang digunakan saat training.

---

## Referensi Bab III

Argonnxx. (2026). *Teen Mental Health Dataset*. Kaggle.
https://www.kaggle.com/datasets/argonnxx/teen-mental-health

Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32.
https://doi.org/10.1023/A:1010933404324

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, *6*(4), 1–21.
https://doi.org/10.1145/2382577.2382579

Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI)*, 1137–1143.

Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. *Proceedings of the 4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining*, 29–39.
