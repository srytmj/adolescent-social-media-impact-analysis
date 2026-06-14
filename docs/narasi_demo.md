# Narasi Demo: Teen Digital Wellbeing Predictor

---

## Pembukaan

Proyek ini adalah sistem prediksi status kesejahteraan digital remaja menggunakan Machine Learning. Kita membangun pipeline lengkap dari awal: mulai dari memuat data, membersihkannya, menganalisis polanya, melatih model, mengevaluasinya, sampai membuat aplikasi interaktif berbasis Streamlit yang bisa langsung digunakan.

Dataset yang kita pakai adalah `Teen_Mental_Health.csv` dari Kaggle, berisi data perilaku digital dan indikator kesehatan mental remaja usia 13 sampai 19 tahun.

---

## Notebook 1 - Data Loading & Understanding

Kita mulai dengan memuat dataset dan memahami strukturnya.

Dataset ini punya 1.200 baris dan 16 kolom. Kolom-kolomnya mencakup informasi demografis seperti usia dan jenis kelamin, perilaku digital seperti durasi media sosial dan screen time sebelum tidur, serta indikator psikologis seperti tingkat stres, kecemasan, dan kecanduan.

Dari `df.info()`, kita bisa langsung lihat tidak ada missing value sama sekali di semua kolom. `df.describe()` juga menunjukkan distribusi data yang cukup merata, misalnya rata-rata penggunaan media sosial sekitar 4,5 jam per hari.

Intinya, data sudah cukup bersih dari awal, tapi kita tetap perlu beberapa langkah pembersihan dan transformasi sebelum bisa dipakai untuk modeling.

---

## Notebook 2 - Data Cleaning & Feature Engineering

Di sini kita melakukan dua hal: membersihkan data dan menciptakan fitur baru.

Untuk pembersihan, ada dua kolom yang kita hapus. Pertama, `platform_usage` - platform mana yang dipakai tidak terlalu relevan dibanding durasi penggunaannya. Kedua, dan ini penting, `depression_label` kita hapus karena berpotensi menyebabkan data leakage. Kolom itu adalah label biner yang merepresentasikan kondisi depresi secara langsung - kalau tetap diikutsertakan, model akan "mencontek" informasi yang seharusnya tidak diketahui saat prediksi di data baru.

Untuk feature engineering, kita tambahkan 5 fitur baru. `total_screen_exposure` menggabungkan durasi sosmed dan screen time sebelum tidur. `sleep_efficiency` mengukur rasio jam tidur terhadap screen time sebelum tidur. `high_social_media_usage` dan `active_lifestyle` adalah flag biner. Dan terakhir, `risk_category` yang mengelompokkan `mental_health_risk_score` ke dalam tiga kategori berdasarkan distribusi kuartil data.

Data teks juga kita encode di sini: Label Encoding untuk `sleep_quality` yang bersifat ordinal, dan One-Hot Encoding untuk `gender`. `drop_first=True` dipakai untuk menghindari dummy variable trap.

---

## Notebook 3 - Exploratory Data Analysis

Setelah data bersih, kita analisis polanya secara visual.

Pertama kita lihat distribusi target `digital_wellbeing_flag`. Count plot menunjukkan sebaran tiga kategori: Healthy, Moderate, dan At Risk.

Lalu kita uji hipotesis: apakah remaja dengan status At Risk memang menghabiskan lebih banyak waktu di media sosial dibanding yang Healthy? Boxplot menunjukkan polanya cukup jelas.

Terakhir, kita buat heatmap korelasi untuk semua fitur numerik. Hasilnya menunjukkan korelasi kuat antara paparan layar, total screen exposure, dan indikator psikologis seperti stres, kecemasan, dan addiction level terhadap risk score.

Pola-pola ini yang jadi dasar kita melanjutkan ke modeling.

---

## Notebook 4 - Data Preprocessing

Sebelum modeling, data perlu dipersiapkan dalam format yang siap pakai.

Target `digital_wellbeing_flag` kita encode: Healthy jadi 0, Moderate jadi 1, At Risk jadi 2. `social_interaction_level` yang belum di-encode juga kita tangani di sini dengan Label Encoding.

Kolom yang kita hapus dari fitur adalah `digital_wellbeing_flag`, `target`, `mental_health_risk_score`, `risk_category`, dan `sleep_quality` - ini untuk menghindari data leakage dari variabel yang derivasinya langsung berkaitan dengan target.

Hasilnya, kita punya 16 fitur dan 1 target. Data ini kita simpan sebagai `preprocessed_data.csv` dalam kondisi belum di-scale, karena proses scaling yang benar harus dilakukan setelah train-test split.

---

## Notebook 5 - Modeling

Ini inti dari projeknya.

Kita pakai pendekatan Supervised Learning dengan algoritma Random Forest Classifier. Alasannya: data kita sudah berlabel, Random Forest tahan overfitting karena menggunakan ensemble dari banyak decision tree, dan ia memberikan feature importance yang berguna untuk interpretasi.

Data dibagi 80:20 untuk training dan testing, dengan `stratify=y` agar distribusi kelas tetap proporsional. Scaler di-fit hanya pada training data, lalu hasilnya di-transform ke test set. Ini penting agar test set tidak "bocor" ke scaler saat training.

Model dilatih dengan `class_weight='balanced'` untuk menangani potensi ketidakseimbangan kelas.

Untuk validasi yang lebih robust, kita pakai K-Fold Cross-Validation dengan K=5. Hasilnya rata-rata akurasi 99.69% dengan gap antara train dan validation hanya 0.0031. Gap sekecil ini membuktikan model tidak overfitting.

Evaluasi pada test set juga menunjukkan akurasi 100%. Ini terdengar terlalu bagus, tapi wajar karena dataset ini memang sintetis - dibuat dengan pola yang deterministik. Buktinya, Fold 1 pada K-Fold menghasilkan 98.44%, bukan 100%, yang membuktikan model tidak sekadar menghafal data.

---

## Notebook 6 - Evaluation

Di sini kita evaluasi model dari berbagai sudut pandang.

Classification Report menunjukkan precision, recall, dan F1-score sempurna di semua kelas. Confusion Matrix memperlihatkan tidak ada satu pun prediksi yang salah pada test set.

Feature Importance menunjukkan fitur-fitur yang paling berpengaruh adalah yang berkaitan dengan perilaku layar: `daily_social_media_hours`, `total_screen_exposure`, dan `screen_time_before_sleep`. Distribusi importance-nya wajar dan relevan secara domain, tidak ada satu fitur yang mendominasi secara tidak normal yang mengindikasikan leakage.

K-Fold CV Summary juga ditampilkan ulang di sini sebagai konfirmasi tambahan - hasilnya konsisten.

---

## Notebook 7 - Deployment (Streamlit)

Notebook terakhir ini men-generate file `app.py` menggunakan `%%writefile`.

Aplikasi Streamlit yang kita buat punya dua bagian utama. Sidebar berisi form input: usia, jenis kelamin, durasi media sosial, kualitas tidur, nilai akademik, kondisi psikologis, dan seterusnya. Main panel menampilkan hasil prediksi beserta probabilitas per kategori dalam bentuk grafik horizontal bar.

Yang menarik, `preprocess_input()` di dalam app mereplikasi seluruh pipeline feature engineering dari Notebook 2: menghitung `total_screen_exposure`, `sleep_efficiency`, flag-flag biner, sampai encoding. Ini memastikan data yang masuk ke model sudah dalam format yang identik dengan saat training.

Setelah prediksi, aplikasi juga menampilkan rekomendasi yang disesuaikan berdasarkan kategori hasilnya, mulai dari saran ringan untuk status Moderate sampai rekomendasi konsultasi profesional untuk At Risk.

Untuk menjalankan aplikasinya cukup dengan:

```bash
streamlit run app.py
```

---

## Penutup

Secara keseluruhan, pipeline yang kita bangun mengikuti alur CRISP-DM: dari data understanding, cleaning, EDA, preprocessing, modeling, evaluation, sampai deployment. Setiap tahap terhubung dan saling bergantung, dan hasilnya adalah aplikasi yang bisa langsung dipakai untuk memprediksi status kesejahteraan digital remaja secara real-time.
