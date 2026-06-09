# Glossary — Istilah Teknis Machine Learning

Dokumen ini berisi penjelasan istilah-istilah teknis yang digunakan
dalam proyek **Teen Digital Wellbeing Predictor**. Glossary ini
ditujukan untuk pembaca yang memiliki latar belakang umum namun
belum familiar dengan terminologi Machine Learning dan Data Science.

---

## Daftar Isi

1. [Model & Algoritma](#a-model--algoritma)
   - [Random Forest](#random-forest)
   - [Decision Tree](#decision-tree)
   - [Ensemble](#ensemble)
   - [Classifier](#classifier)
   - [class_weight='balanced'](#class_weightbalanced)

2. [Metrik Evaluasi](#b-metrik-evaluasi)
   - [Accuracy](#accuracy)
   - [Precision](#precision)
   - [Recall](#recall)
   - [F1-Score](#f1-score)
   - [Macro Average](#macro-average)
   - [Weighted Average](#weighted-average)
   - [Support](#support)
   - [Confusion Matrix](#confusion-matrix)
   - [False Positive](#false-positive)
   - [False Negative](#false-negative)
   - [True Positive](#true-positive)

3. [Validasi & Splitting](#c-validasi--splitting)
   - [Train-Test Split](#train-test-split)
   - [Hold-out](#hold-out)
   - [K-Fold Cross-Validation](#k-fold-cross-validation)
   - [Stratified K-Fold](#stratified-k-fold)
   - [Fold](#fold)
   - [CV](#cv)
   - [Gap Train-Val](#gap-train-val)
   - [Overfitting](#overfitting)
   - [Underfitting](#underfitting)
   - [Generalisasi](#generalisasi)

4. [Preprocessing & Fitur](#d-preprocessing--fitur)
   - [Feature Engineering](#feature-engineering)
   - [Feature Importance](#feature-importance)
   - [Data Leakage](#data-leakage)
   - [StandardScaler](#standardscaler)
   - [Label Encoding](#label-encoding)
   - [One-Hot Encoding](#one-hot-encoding)
   - [Stratify](#stratify)
   - [Imbalanced Class](#imbalanced-class)

5. [Dataset](#e-dataset)
   - [Dataset Sintetis](#dataset-sintetis)
   - [Deterministik](#deterministik)
   - [Ground Truth](#ground-truth)
   - [Distribusi Kelas](#distribusi-kelas)

6. [Referensi](#referensi)
   - [Buku & Artikel Ilmiah](#buku--artikel-ilmiah)
   - [Dokumentasi Resmi](#dokumentasi-resmi)
   - [Artikel & Tutorial Online](#artikel--tutorial-online)
   - [Referensi Domain Kesehatan Mental Remaja](#referensi-domain-kesehatan-mental-remaja)

---

## A. Model & Algoritma

### Random Forest
Algoritma Machine Learning yang bekerja dengan membangun banyak 
*decision tree* secara bersamaan, lalu menggabungkan hasil prediksi 
dari seluruh pohon tersebut (voting mayoritas). Pendekatan ini 
disebut *ensemble learning* dan terbukti lebih akurat serta lebih 
tahan terhadap overfitting dibandingkan satu decision tree tunggal.

### Decision Tree
Model Machine Learning berbentuk pohon keputusan yang memisahkan 
data berdasarkan serangkaian aturan logis dari fitur-fiturnya. 
Setiap cabang mewakili kondisi tertentu, dan setiap daun 
mewakili hasil prediksi akhir.

### Ensemble
Teknik dalam Machine Learning yang menggabungkan banyak model 
sederhana menjadi satu model yang lebih kuat dan lebih stabil. 
Prinsipnya: banyak model lemah yang bekerja bersama lebih baik 
daripada satu model kuat yang bekerja sendiri.

### Classifier
Model Machine Learning yang tugasnya memprediksi kategori atau 
kelas dari suatu data — bukan memprediksi angka kontinu. Contoh 
dalam proyek ini: mengklasifikasikan remaja ke dalam kategori 
*Healthy*, *Moderate*, atau *At Risk*.

### class_weight='balanced'
Parameter pada model yang secara otomatis menyesuaikan bobot 
tiap kelas berdasarkan jumlah sampelnya. Kelas dengan jumlah 
data lebih sedikit (minoritas) akan diberi bobot lebih besar, 
sehingga model tidak mengabaikan kelas tersebut saat training.

---

## B. Metrik Evaluasi

### Accuracy
Proporsi prediksi yang benar dari seluruh total prediksi. 
Rumus: (jumlah prediksi benar) / (total data). Metrik ini 
mudah dipahami, namun kurang representatif jika jumlah data 
antar kelas tidak seimbang.

### Precision
Dari semua data yang diprediksi sebagai kelas X, berapa persen 
yang benar-benar kelas X. Precision tinggi berarti model jarang 
melakukan *false positive* — jarang salah menuduh.

Contoh: Precision *At Risk* = 1.00 artinya setiap kali model 
memprediksi remaja sebagai *At Risk*, prediksi tersebut selalu 
benar.

### Recall
Dari semua data yang sebenarnya kelas X, berapa persen yang 
berhasil terdeteksi oleh model. Recall tinggi berarti model 
jarang melewatkan kasus yang seharusnya terdeteksi (*false 
negative* rendah).

Contoh: Recall *At Risk* = 1.00 artinya model berhasil 
mendeteksi seluruh remaja yang benar-benar *At Risk* tanpa 
ada yang terlewat.

### F1-Score
Rata-rata harmonik dari Precision dan Recall. Digunakan saat 
kita ingin keseimbangan antara keduanya, terutama pada dataset 
dengan distribusi kelas yang tidak seimbang. Nilai F1 mendekati 
1.00 berarti model memiliki Precision dan Recall yang sama-sama 
tinggi.

### Macro Average
Rata-rata metrik (Precision, Recall, F1) yang dihitung dengan 
memberikan bobot yang sama pada setiap kelas, tanpa 
mempertimbangkan jumlah sampel per kelas. Cocok digunakan 
ketika kita ingin mengevaluasi performa model secara adil 
pada semua kelas, termasuk kelas minoritas.

### Weighted Average
Rata-rata metrik yang dihitung dengan mempertimbangkan 
proporsi jumlah sampel per kelas. Kelas dengan jumlah data 
lebih banyak mendapat bobot lebih besar dalam perhitungan.

### Support
Jumlah data aktual per kelas yang terdapat pada test set. 
Angka ini mencerminkan distribusi kelas pada data uji dan 
berguna untuk memahami konteks nilai metrik lainnya.

### Confusion Matrix
Tabel yang menampilkan distribusi hasil prediksi model 
dibandingkan label aslinya. Baris mewakili kelas aktual, 
kolom mewakili kelas prediksi. Diagonal utama (kiri atas 
ke kanan bawah) menunjukkan prediksi yang benar, sedangkan 
nilai di luar diagonal menunjukkan kesalahan prediksi.

### False Positive
Kondisi di mana model memprediksi suatu data sebagai positif 
(misalnya *At Risk*), padahal kenyataannya data tersebut 
negatif (misalnya *Healthy*). Disebut juga "salah menuduh".

### False Negative
Kondisi di mana model memprediksi suatu data sebagai negatif 
(misalnya *Healthy*), padahal kenyataannya data tersebut 
positif (misalnya *At Risk*). Disebut juga "melewatkan kasus". 
Dalam konteks kesehatan mental, false negative lebih berbahaya 
daripada false positive.

### True Positive
Kondisi di mana model memprediksi suatu data sebagai positif 
dan kenyataannya memang positif — prediksi yang benar.

---

## C. Validasi & Splitting

### Train-Test Split
Teknik membagi dataset menjadi dua bagian: data latih (*training 
set*) untuk melatih model, dan data uji (*test set*) untuk 
mengevaluasi performa model pada data yang belum pernah dilihat 
sebelumnya. Proyek ini menggunakan proporsi 80:20.

### Hold-out
Data yang sengaja dipisahkan dari proses training dan hanya 
digunakan sekali di akhir untuk evaluasi final. Dalam proyek 
ini, 20% data dijadikan hold-out test set.

### K-Fold Cross-Validation
Teknik validasi model yang lebih robust dibandingkan train-test 
split biasa. Data training dibagi menjadi K lipatan (*fold*) 
yang sama besar. Model dilatih sebanyak K kali — setiap kali 
menggunakan K-1 fold untuk training dan 1 fold untuk validasi. 
Hasil akhir adalah rata-rata performa dari K percobaan tersebut.

Proyek ini menggunakan K=5, artinya model dilatih dan divalidasi 
sebanyak 5 kali dengan pembagian data yang berbeda-beda.

### Stratified K-Fold
Varian K-Fold Cross-Validation yang menjaga proporsi kelas 
target tetap seimbang di setiap fold. Penting digunakan ketika 
distribusi kelas tidak seimbang agar setiap fold merepresentasikan 
keseluruhan dataset dengan baik.

### Fold
Satu bagian atau lipatan data dalam K-Fold Cross-Validation. 
Dengan K=5, data dibagi menjadi 5 fold yang masing-masing 
berisi 20% dari total data training.

### CV
Singkatan dari *Cross-Validation* — teknik validasi model dengan 
membagi data ke beberapa subset untuk mendapatkan estimasi 
performa yang lebih andal.

### Gap Train-Val
Selisih antara performa model pada data training dan data 
validasi. Gap kecil (< 0.05) menandakan model tidak overfitting 
dan mampu generalisasi dengan baik. Dalam proyek ini, gap 
Train-Val = 0.0031.

### Overfitting
Kondisi di mana model terlalu "hafal" pola dari data training, 
termasuk noise dan keunikan spesifik data tersebut, sehingga 
performanya buruk ketika dihadapkan pada data baru. Ditandai 
dengan gap besar antara Train Accuracy dan Validation Accuracy.

### Underfitting
Kondisi di mana model terlalu sederhana sehingga tidak mampu 
menangkap pola yang ada dalam data. Ditandai dengan performa 
rendah baik pada data training maupun data validasi.

### Generalisasi
Kemampuan model untuk bekerja dengan baik pada data baru yang 
belum pernah dilihat selama proses training. Model yang 
generalisasi baik berarti ia benar-benar belajar pola umum, 
bukan sekadar menghafal data training.

---

## D. Preprocessing & Fitur

### Feature Engineering
Proses membuat fitur-fitur baru dari fitur yang sudah ada 
dengan tujuan meningkatkan kemampuan model dalam menangkap 
pola. Contoh dalam proyek ini: `total_screen_exposure` dibuat 
dari penjumlahan `daily_social_media_hours` dan 
`screen_time_before_sleep`.

### Feature Importance
Skor yang dihasilkan oleh model Random Forest untuk menunjukkan 
seberapa besar kontribusi relatif setiap fitur dalam proses 
pengambilan keputusan. Fitur dengan importance tinggi berarti 
fitur tersebut paling sering dan paling berpengaruh dalam 
memisahkan kelas target.

### Data Leakage
Kondisi berbahaya di mana informasi dari variabel target secara 
tidak sengaja "bocor" ke dalam fitur prediktor. Akibatnya, 
model terlihat memiliki akurasi sangat tinggi (bahkan 100%), 
padahal performa tersebut tidak mencerminkan kemampuan model 
yang sesungguhnya pada data dunia nyata.

### StandardScaler
Teknik standarisasi fitur numerik agar memiliki rata-rata 0 
dan standar deviasi 1. Tujuannya agar fitur dengan skala besar 
(misalnya skor 1–100) tidak mendominasi fitur dengan skala 
kecil (misalnya 0–1) dalam proses training model.

### Label Encoding
Teknik mengubah variabel kategorikal yang bersifat ordinal 
(memiliki urutan) menjadi angka berurutan. Contoh: 
`social_interaction_level` diubah menjadi low=0, medium=1, 
high=2.

### One-Hot Encoding
Teknik mengubah variabel kategorikal yang bersifat nominal 
(tidak memiliki urutan) menjadi beberapa kolom biner (0 atau 1), 
satu kolom per kategori. Contoh: `gender` diubah menjadi 
kolom `gender_male` dengan nilai 1 (laki-laki) atau 0 
(perempuan).

### Stratify
Parameter pada fungsi train-test split yang memastikan 
proporsi kelas target tetap sama di data training maupun 
data testing, sesuai dengan distribusi asli di dataset.

### Imbalanced Class
Kondisi di mana jumlah data antar kelas dalam dataset tidak 
seimbang. Contoh dalam proyek ini: kelas *Moderate* memiliki 
149 data pada test set, sedangkan kelas *At Risk* hanya 30 data.

---

## E. Dataset

### Dataset Sintetis
Dataset yang dibuat secara artifisial menggunakan aturan atau 
formula tertentu, bukan berasal dari pengukuran atau observasi 
dunia nyata. Dataset sintetis cenderung memiliki pola yang 
lebih bersih dan konsisten, sehingga model Machine Learning 
dapat mencapai akurasi lebih tinggi dibandingkan pada data 
dunia nyata.

### Deterministik
Sifat data atau sistem di mana output dapat diprediksi secara 
pasti dari input yang diberikan, tanpa ada unsur keacakan 
atau noise. Dataset sintetis bersifat deterministik karena 
dibuat dengan aturan yang tetap.

### Ground Truth
Label atau jawaban yang benar dari setiap data, yang dijadikan 
acuan untuk mengevaluasi prediksi model. Dalam proyek ini, 
ground truth adalah nilai kolom `digital_wellbeing_flag` 
yang sudah ada di dataset.

### Distribusi Kelas
Proporsi jumlah data per kelas dalam dataset. Mengetahui 
distribusi kelas penting untuk memilih metrik evaluasi yang 
tepat dan memastikan model tidak bias terhadap kelas mayoritas.

---

*Glossary ini akan diperbarui seiring dengan perkembangan proyek.*

---

## Referensi

### Buku & Artikel Ilmiah

1. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
   https://doi.org/10.1023/A:1010933404324

2. Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for 
   Accuracy Estimation and Model Selection*. Proceedings of the 14th 
   International Joint Conference on Artificial Intelligence (IJCAI), 
   1137–1143.

3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). 
   *The Elements of Statistical Learning: Data Mining, Inference, 
   and Prediction* (2nd ed.). Springer.
   https://hastie.su.domains/ElemStatLearn/

4. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, 
   Keras, and TensorFlow* (3rd ed.). O'Reilly Media.

5. Provost, F., & Fawcett, T. (2013). *Data Science for Business*. 
   O'Reilly Media.

---

### Dokumentasi Resmi

6. Scikit-learn Developers. (2024). *RandomForestClassifier*. 
   Scikit-learn Documentation.
   https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

7. Scikit-learn Developers. (2024). *Cross-validation: evaluating 
   estimator performance*. Scikit-learn Documentation.
   https://scikit-learn.org/stable/modules/cross_validation.html

8. Scikit-learn Developers. (2024). *StratifiedKFold*. 
   Scikit-learn Documentation.
   https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

9. Scikit-learn Developers. (2024). *StandardScaler*. 
   Scikit-learn Documentation.
   https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

10. Scikit-learn Developers. (2024). *Feature importances with a 
    forest of trees*. Scikit-learn Documentation.
    https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

---

### Artikel & Tutorial Online

11. Brownlee, J. (2020). *A Gentle Introduction to k-fold 
    Cross-Validation*. Machine Learning Mastery.
    https://machinelearningmastery.com/k-fold-cross-validation/

12. Brownlee, J. (2020). *How to Fix k-Fold Cross-Validation 
    for Imbalanced Classification*. Machine Learning Mastery.
    https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/

13. Brownlee, J. (2021). *Random Oversampling and Undersampling 
    for Imbalanced Classification*. Machine Learning Mastery.
    https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/

14. Towards Data Science. (2021). *Understanding Feature Importance 
    and How to Implement it in Python*.
    https://towardsdatascience.com/understanding-feature-importance-and-how-to-implement-it-in-python-ff0287b20285

15. Towards Data Science. (2020). *Data Leakage in Machine Learning*.
    https://towardsdatascience.com/data-leakage-in-machine-learning-6161c167e8a8

---

### Referensi Domain Kesehatan Mental Remaja

16. World Health Organization. (2021). *Adolescent Mental Health*. 
    WHO Fact Sheets.
    https://www.who.int/news-room/fact-sheets/detail/adolescent-mental-health

17. Twenge, J. M., & Campbell, W. K. (2019). *Media Use Is Linked 
    to Lower Psychological Well-Being: Evidence from Three Datasets*. 
    Psychiatric Quarterly, 90(2), 311–331.
    https://doi.org/10.1007/s11126-019-09630-7

18. Kementerian Kesehatan Republik Indonesia. (2021). 
    *Pedoman Dukungan Kesehatan Jiwa dan Psikososial pada Pandemi 
    COVID-19*. Kemenkes RI.
    https://www.kemkes.go.id

19. Twenge, J. M. (2017). *iGen: Why Today's Super-Connected Kids 
    Are Growing Up Less Rebellious, More Tolerant, Less Happy — 
    and Completely Unprepared for Adulthood*. Atria Books.

20. American Psychological Association. (2023). 
    *Social Media and Youth Mental Health: The U.S. Surgeon 
    General's Advisory*.
    https://www.apa.org/topics/social-media-internet/social-media-youth-mental-health