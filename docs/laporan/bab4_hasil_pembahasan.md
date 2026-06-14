# BAB IV HASIL DAN PEMBAHASAN

---

## 4.1 Hasil Penelitian

### 4.1.1 Deskripsi Dataset

Dataset yang digunakan dalam penelitian ini adalah *Teen Mental Health Dataset* yang diperoleh dari platform Kaggle. Dataset terdiri dari 1.200 sampel data remaja berusia 13 hingga 19 tahun dengan 16 atribut awal yang mencakup tiga kelompok variabel: informasi demografis (usia dan jenis kelamin), perilaku digital (durasi penggunaan media sosial harian, *screen time* sebelum tidur, tingkat kecanduan), serta indikator kesehatan mental (tingkat stres, kecemasan, dan skor risiko kesehatan mental). Tidak ditemukan *missing value* maupun data duplikat pada dataset, sehingga seluruh 1.200 sampel dapat digunakan dalam proses analisis dan pemodelan.

Variabel target yang diprediksi adalah `digital_wellbeing_flag` dengan tiga kategori kelas sebagai berikut.

| Kelas | Label | Jumlah Sampel | Proporsi |
|---|---|---|---|
| Healthy | 0 | 364 | 30.3% |
| Moderate | 1 | 746 | 62.2% |
| At Risk | 2 | 90 | 7.5% |

### 4.1.2 Hasil Feature Engineering

Pada tahap *Data Cleaning and Feature Engineering*, dilakukan penghapusan dua kolom yaitu `platform_usage` dan `depression_label`. Kolom `depression_label` secara khusus dihapus karena teridentifikasi sebagai potensi *data leakage*, yakni variabel yang secara langsung merepresentasikan kondisi depresi remaja dan beririsan dengan konstruksi target `digital_wellbeing_flag`.

Selanjutnya, lima fitur baru diturunkan dari fitur yang telah ada sebagaimana tercantum dalam Tabel 4.2 berikut.

| Fitur Baru | Definisi | Justifikasi |
|---|---|---|
| `total_screen_exposure` | Durasi sosmed + *screen time* sebelum tidur | Representasi total paparan layar harian |
| `sleep_efficiency` | Jam tidur / (*screen time* sebelum tidur + 1) | Rasio kualitas tidur terhadap gangguan layar |
| `high_social_media_usage` | Flag 1 jika durasi sosmed > 5 jam/hari | Penanda kebiasaan penggunaan berlebih |
| `active_lifestyle` | Flag 1 jika aktivitas fisik >= 1 jam/hari | Penanda gaya hidup aktif |
| `risk_category` | Kategorisasi `mental_health_risk_score` berdasarkan kuartil | Segmentasi risiko berbasis distribusi aktual |

Setelah proses *encoding* (Label Encoding untuk `sleep_quality`, One-Hot Encoding untuk `gender`) dan penghapusan kolom yang berpotensi *leakage* pada tahap *preprocessing*, jumlah fitur final yang digunakan sebagai input model adalah **16 fitur**.

### 4.1.3 Hasil Pemodelan

Data dibagi dengan proporsi 80:20 menggunakan stratifikasi kelas (`stratify=y`) untuk memastikan distribusi kelas tetap proporsional pada kedua subset. *StandardScaler* di-*fit* hanya pada data latih (*training set*) guna menghindari *data leakage* dari data uji ke proses normalisasi. Rincian pembagian data tercantum pada Tabel 4.3.

| Subset | Jumlah Sampel |
|---|---|
| Training Set | 960 |
| Test Set | 240 |

Model yang digunakan adalah **Random Forest Classifier** dengan konfigurasi `n_estimators=100`, `random_state=42`, dan `class_weight='balanced'` untuk menangani potensi ketidakseimbangan distribusi kelas.

#### 4.1.3.1 Hasil K-Fold Cross-Validation

Validasi model dilakukan menggunakan *Stratified K-Fold Cross-Validation* dengan K=5. Hasil selengkapnya disajikan pada Tabel 4.4.

| Metrik | Rata-rata | Standar Deviasi |
|---|---|---|
| Accuracy | 0.9969 | ±0.0063 |
| F1-Score (Macro) | 0.9947 | ±0.0105 |
| Precision (Macro) | 0.9984 | ±0.0033 |
| Recall (Macro) | 0.9917 | ±0.0167 |
| Train Accuracy | 1.0000 | - |
| Gap Train-Validation | 0.0031 | - |

Akurasi per fold pada proses validasi silang adalah sebagai berikut: Fold 1: 98.44%, Fold 2: 100%, Fold 3: 100%, Fold 4: 100%, Fold 5: 100%. Gap antara *train accuracy* dan *validation accuracy* sebesar 0.0031 berada jauh di bawah ambang batas 0.05, mengindikasikan bahwa model tidak mengalami *overfitting*.

#### 4.1.3.2 Hasil Evaluasi pada Test Set

Evaluasi akhir model dilakukan pada *test set* (hold-out 20%) yang menghasilkan *classification report* sebagaimana tercantum pada Tabel 4.5.

| Kelas | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Healthy (0) | 1.00 | 1.00 | 1.00 | 61 |
| Moderate (1) | 1.00 | 1.00 | 1.00 | 149 |
| At Risk (2) | 1.00 | 1.00 | 1.00 | 30 |
| **Macro Avg** | **1.00** | **1.00** | **1.00** | **240** |
| **Weighted Avg** | **1.00** | **1.00** | **1.00** | **240** |

*Confusion Matrix* menunjukkan bahwa seluruh 240 sampel pada *test set* berhasil diklasifikasikan dengan benar tanpa satu pun kesalahan prediksi.

#### 4.1.3.3 Hasil Feature Importance

Analisis *feature importance* dilakukan untuk mengidentifikasi fitur-fitur yang paling berpengaruh dalam pengambilan keputusan model. Lima fitur dengan nilai *importance* tertinggi disajikan pada Tabel 4.6.

| Peringkat | Fitur | Keterangan |
|---|---|---|
| 1 | `daily_social_media_hours` | Durasi harian penggunaan media sosial |
| 2 | `total_screen_exposure` | Total paparan layar harian (fitur turunan) |
| 3 | `screen_time_before_sleep` | *Screen time* sebelum tidur |
| 4 | `sleep_efficiency` | Rasio kualitas tidur (fitur turunan) |
| 5 | `stress_level` | Tingkat stres |

---

## 4.2 Pembahasan Hasil

### 4.2.1 Analisis Performa Model

Model Random Forest Classifier yang dikembangkan mencapai akurasi 100% pada *test set* dan rata-rata 99.69% pada *K-Fold Cross-Validation*. Capaian ini konsisten dengan karakteristik dataset yang digunakan, yaitu dataset sintetis (*synthetic dataset*) yang dibentuk berdasarkan aturan deterministik. Dataset sintetis memiliki pola hubungan antara fitur dan target yang jauh lebih konsisten dibandingkan data empiris dari dunia nyata karena tidak adanya *noise* sistemik yang umumnya hadir pada data observasional.

Bukti bahwa model tidak sekadar menghafal data (*memorization*) ditunjukkan oleh dua indikasi. Pertama, Fold 1 pada *K-Fold CV* menghasilkan akurasi 98.44%, bukan 100%, yang membuktikan adanya variasi generalisasi antar-fold. Kedua, *gap* antara *train accuracy* (100%) dan *validation accuracy* (99.69%) hanya sebesar 0.0031, jauh di bawah ambang batas *overfitting* yang umum digunakan (0.05) sebagaimana dinyatakan oleh Ying (2019).

### 4.2.2 Interpretasi Feature Importance

Hasil analisis *feature importance* menunjukkan bahwa fitur-fitur yang berkaitan langsung dengan perilaku penggunaan layar mendominasi kontribusi terhadap prediksi model. Fitur `daily_social_media_hours` menempati posisi pertama, diikuti oleh `total_screen_exposure` yang merupakan fitur turunan hasil *feature engineering*. Hal ini mengkonfirmasi relevansi fitur yang dikonstruksi secara manual dalam meningkatkan daya prediksi model.

Temuan ini selaras dengan berbagai studi yang telah mendokumentasikan hubungan antara intensitas penggunaan media sosial dengan kondisi kesehatan mental remaja. Twenge et al. (2018) menemukan korelasi negatif yang signifikan antara peningkatan *screen time* dan indikator kesejahteraan psikologis pada remaja Amerika Serikat.

Fitur `sleep_efficiency` yang berada di peringkat keempat turut memperkuat temuan Levenson et al. (2017) mengenai hubungan antara penggunaan media sosial sebelum tidur dengan kualitas tidur yang buruk pada remaja, yang pada akhirnya berkontribusi terhadap memburuknya kondisi kesehatan mental secara keseluruhan.

Distribusi *feature importance* yang proporsional dan tidak didominasi oleh satu fitur tunggal secara tidak wajar juga menjadi konfirmasi tambahan bahwa tidak terdapat *data leakage* yang tersembunyi dalam pipeline pemodelan. Hal ini penting mengingat kolom `depression_label` dan `mental_health_risk_score` telah dieliminasi pada tahap awal sebagai langkah preventif terhadap kebocoran informasi.

### 4.2.3 Pemilihan Algoritma Random Forest

Pemilihan algoritma Random Forest didasarkan pada beberapa pertimbangan teknis. Sebagai metode *ensemble* berbasis *bagging* dari sejumlah *decision tree*, Random Forest memiliki mekanisme bawaan untuk mengurangi *variance* dan mencegah *overfitting* (Breiman, 2001). Penggunaan parameter `class_weight='balanced'` juga memungkinkan model untuk memberikan bobot proporsional terhadap kelas minoritas (*At Risk*) yang hanya memiliki 90 sampel dari total 1.200 data, sehingga mengurangi bias prediksi terhadap kelas mayoritas (*Moderate*).

Dibandingkan dengan algoritma klasifikasi lain seperti *Support Vector Machine* (SVM) atau *Logistic Regression*, Random Forest menawarkan keunggulan berupa kemampuan menangani hubungan non-linear antar fitur tanpa memerlukan asumsi distribusi data, serta kemampuan menghasilkan *feature importance* yang dapat diinterpretasikan secara langsung untuk keperluan analisis domain (Liaw & Wiener, 2002).

### 4.2.4 Validitas Pipeline Preprocessing

Salah satu aspek kritis dalam pipeline penelitian ini adalah keputusan untuk melakukan *fit* StandardScaler hanya pada data latih, kemudian menerapkan transformasi yang sama (*transform*) pada data uji. Pendekatan ini secara eksplisit mencegah terjadinya *data leakage* pada tahap normalisasi, di mana informasi statistik dari data uji (nilai rata-rata dan standar deviasi) tidak ikut mempengaruhi proses pelatihan model. Kerangka ini konsisten dengan praktik terbaik yang direkomendasikan oleh Kaufman et al. (2012) dalam konteks evaluasi model yang valid secara metodologis.

---

## 4.3 Screenshot Dashboard

Aplikasi prediksi kesejahteraan digital remaja diimplementasikan menggunakan framework Streamlit dan dapat dijalankan secara lokal melalui perintah `streamlit run app.py`. Antarmuka aplikasi terdiri dari dua panel utama: panel *sidebar* untuk input data remaja dan panel utama untuk menampilkan hasil prediksi beserta rekomendasi intervensi.

Panel *sidebar* memuat form input yang mencakup dua belas variabel meliputi usia, jenis kelamin, durasi penggunaan media sosial harian, *screen time* sebelum tidur, durasi tidur, kualitas tidur, nilai akademik (*GPA*), aktivitas fisik, tingkat interaksi sosial, tingkat stres, tingkat kecemasan, dan tingkat kecanduan media sosial. Seluruh variabel input direpresentasikan dalam bentuk *slider* atau *selectbox* yang intuitif untuk digunakan.

Panel utama menampilkan tiga komponen visualisasi setelah tombol prediksi ditekan. Pertama, badge status hasil prediksi (*Healthy*, *Moderate*, atau *At Risk*) dengan kode warna yang berbeda untuk setiap kategori (hijau, oranye, merah). Kedua, grafik *horizontal bar chart* yang menampilkan probabilitas prediksi untuk masing-masing kategori, memungkinkan pengguna untuk memahami tingkat keyakinan model. Ketiga, panel rekomendasi yang berisi langkah-langkah intervensi yang disesuaikan secara dinamis berdasarkan kategori hasil prediksi.

---

*[TEMPAT SCREENSHOT DASHBOARD INTERAKTIF]*

*(Ganti area ini dengan gambar tangkapan layar visualisasi dashboard yang sesungguhnya)*

---

**Gambar 4.1** Screenshot Tampilan Utama Dashboard Interaktif *Teen Digital Wellbeing Predictor*
