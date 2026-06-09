[Kembali](index.md)

# BAB I
# PENDAHULUAN

## 1.1 Latar Belakang

Perkembangan teknologi digital yang pesat dalam dua dekade terakhir telah mengubah secara fundamental cara manusia berinteraksi, berkomunikasi, dan mengakses informasi. Media sosial sebagai salah satu produk utama revolusi digital kini telah menjadi bagian yang tidak terpisahkan dari kehidupan sehari-hari, khususnya bagi generasi muda. Dalam kondisi ideal, pemanfaatan teknologi digital seharusnya memberikan dampak positif berupa kemudahan akses informasi, perluasan jaringan sosial, dan peningkatan produktivitas. Namun kenyataan di lapangan menunjukkan adanya kesenjangan yang signifikan antara potensi manfaat tersebut dengan dampak negatif yang ditimbulkan, terutama pada kelompok remaja yang sedang berada dalam fase perkembangan psikologis yang rentan.
Di tingkat global, penggunaan media sosial di kalangan remaja terus meningkat secara masif. Indonesia sendiri mencatat angka yang mengkhawatirkan: sebanyak 180 juta penduduk Indonesia aktif menggunakan media sosial dengan rata-rata durasi penggunaan mencapai 3 jam per hari [(Komdigi, 2025)](https://kumparan.com/kumparannews/180-juta-orang-indonesia-main-medsos-rata-rata-habiskan-3-jam-sehari-275I28pbgaK). Angka ini menempatkan Indonesia sebagai salah satu negara dengan tingkat konsumsi media sosial tertinggi di dunia. Lebih jauh, Kementerian Komunikasi dan Digital Republik Indonesia (Komdigi) mencatat bahwa anak-anak dan remaja merupakan kelompok pengguna yang paling rentan terhadap dampak negatif ruang digital, mulai dari paparan konten berbahaya, cyberbullying, hingga gangguan kesehatan mental akibat ketergantungan terhadap platform digital [(Komdigi, 2025)](https://www.komdigi.go.id/berita/artikel/detail/komitmen-pemerintah-melindungi-anak-di-ruang-digital).
Fenomena ini menunjukkan adanya kesenjangan serius antara kondisi ideal penggunaan teknologi digital yang sehat dengan realita perilaku digital remaja Indonesia saat ini. Berbagai penelitian menunjukkan bahwa penggunaan media sosial yang berlebihan pada remaja berkorelasi dengan meningkatnya tingkat stres, kecemasan, depresi, gangguan pola tidur, serta penurunan prestasi akademik [(Twenge & Campbell, 2019)](https://doi.org/10.1007/s11126-019-09630-7). Meskipun demikian, deteksi dini terhadap risiko gangguan kesehatan mental akibat perilaku digital yang tidak sehat masih sangat terbatas, baik dari sisi ketersediaan alat deteksi yang mudah diakses maupun dari sisi kesadaran masyarakat dan tenaga pendidik.
Dalam konteks inilah pendekatan Machine Learning menjadi relevan sebagai solusi teknologi yang dapat membantu proses deteksi dini secara otomatis, objektif, dan terukur. Dengan memanfaatkan data perilaku digital dan indikator psikologis remaja, model Machine Learning dapat dilatih untuk mengklasifikasikan status kesejahteraan digital remaja ke dalam kategori tertentu, sehingga memungkinkan intervensi yang lebih cepat dan tepat sasaran.
Berdasarkan latar belakang tersebut, penelitian ini mengembangkan sistem prediksi status kesejahteraan digital remaja menggunakan algoritma Random Forest Classifier dengan kerangka kerja CRISP-DM (Cross-Industry Standard Process for Data Mining). Dataset yang digunakan adalah Teen Mental Health Dataset yang tersedia secara publik di platform Kaggle [(Argonnxx, 2026)](https://www.kaggle.com/datasets/argonnxx/teen-mental-health), yang berisi 1.200 data rekaman perilaku digital dan indikator kesehatan mental remaja usia 13 hingga 19 tahun. Hasil prediksi sistem ini diharapkan dapat menjadi acuan bagi orang tua, tenaga pendidik, maupun konselor dalam memberikan intervensi yang tepat kepada remaja yang berisiko.

---

## 1.2 Rumusan Masalah

Berdasarkan latar belakang yang telah diuraikan, rumusan masalah dalam penelitian ini adalah sebagai berikut:
1. Bagaimana karakteristik dataset perilaku digital dan kesehatan mental remaja yang digunakan dalam penelitian ini?
2. Bagaimana proses preprocessing dan feature engineering  tepat untuk mempersiapkan data sebelum pemodelan?
3. Bagaimana performa model Random Forest Classifier dalam memprediksi status kesejahteraan digital remaja berdasarkan metrik akurasi, precision, recall, dan F1-Score?
4. Bagaimana konsistensi dan keandalan model dibuktikan melalui K-Fold Cross-Validation (K=5)?
5. Bagaimana sistem prediksi yang dibangun dapat diimplementasikan dalam bentuk aplikasi yang mudah digunakan oleh pengguna umum?

---

## 1.3 Tujuan Penelitian

Berdasarkan rumusan masalah di atas, tujuan penelitian ini adalah:
1. Mengetahui dan memahami karakteristik dataset perilaku digital dan kesehatan mental remaja melalui analisis eksploratif (EDA).
2. Merancang proses preprocessing dan feature engineering  tepat untuk menghasilkan data yang siap digunakan dalam pemodelan Machine Learning.
3. Mengukur dan menganalisis performa model Random Forest Classifier dalam mengklasifikasikan status kesejahteraan digital remaja ke dalam kategori Healthy, Moderate, dan At Risk.
4. Membuktikan konsistensi dan keandalan model melalui validasi K-Fold Cross-Validation dengan K=5.
5. Mengimplementasikan model prediksi dalam bentuk aplikasi web interaktif berbasis Streamlit yang dapat digunakan untuk prediksi data baru secara real-time beserta rekomendasi tindakan yang relevan.

---

## 1.4 Manfaat Penelitian

Penelitian ini diharapkan memberikan manfaat sebagai berikut:

### A. Manfaat Praktis

1. Bagi Orang Tua dan Tenaga Pendidik Sistem prediksi yang dibangun dapat digunakan sebagai alat bantu deteksi dini untuk mengidentifikasi remaja yang berisiko mengalami gangguan kesehatan mental akibat perilaku digital yang tidak sehat, sehingga intervensi dapat dilakukan lebih cepat dan tepat sasaran.

2. Bagi Konselor dan Psikolog Sekolah Hasil prediksi beserta rekomendasi yang dihasilkan sistem dapat menjadi data pendukung dalam proses konseling dan penyusunan program pembinaan kesehatan mental remaja.

3. Bagi Pengambil Kebijakan Temuan penelitian ini dapat menjadi referensi bagi instansi terkait seperti Kementerian Komunikasi dan Digital (Komdigi) dan Kementerian Pendidikan dalam merancang kebijakan literasi digital yang lebih efektif bagi remaja.

### B. Manfaat Akademis

1. Memberikan kontribusi pada pengembangan penerapan algoritma Random Forest Classifier dalam domain kesehatan mental dan perilaku digital remaja.

2. Menjadi referensi bagi penelitian selanjutnya yang berkaitan dengan deteksi dini kesehatan mental berbasis Machine Learning.

3. Memperkenalkan penerapan kerangka kerja CRISP-DM secara menyeluruh mulai dari pemahaman data hingga deployment aplikasi prediksi.

---

## 1.5 Batasan Penelitian

Agar pembahasan dalam penelitian ini tetap fokus dan terarah,
ditetapkan batasan-batasan penelitian sebagai berikut:

1. Dataset yang digunakan adalah Teen Mental Health Dataset (Argonnxx, 2026) yang diambil dari platform Kaggle, terdiri dari 1.200 baris data dengan 16 variabel, mencakup remaja berusia 13 hingga 19 tahun.
2. Dataset yang digunakan merupakan dataset sintetis (data buatan) yang dibuat dengan aturan deterministik, sehingga hasil penelitian ini belum tentu sepenuhnya merepresentasikan kondisi populasi remaja di dunia nyata.
3. Algoritma Machine Learning yang digunakan terbatas pada Random Forest Classifier dengan pendekatan Supervised Learning untuk tugas klasifikasi tiga kelas (Healthy, Moderate, At Risk).
4. Validasi model dilakukan menggunakan K-Fold Cross-Validation dengan K=5 dan train-test split dengan proporsi 80:20.
5. Variabel target dalam penelitian ini adalah digital_wellbeing_flag dengan tiga kategori: Healthy, Moderate, dan At Risk.
6. Implementasi sistem prediksi dibatasi pada aplikasi web berbasis Streamlit yang berjalan secara lokal, tanpa integrasi ke sistem database eksternal atau layanan cloud production.
7. Rekomendasi yang dihasilkan sistem bersifat umum dan tidak menggantikan diagnosis klinis dari tenaga kesehatan mental yang berkompeten.


---

## Referensi Bab I

Argonnxx. (2024). *Teen Mental Health Dataset*. Kaggle.
https://www.kaggle.com/datasets/argonnxx/teen-mental-health

Kementerian Komunikasi dan Digital Republik Indonesia. (2024).
*Komitmen Pemerintah Melindungi Anak di Ruang Digital*. Komdigi.
https://www.komdigi.go.id/berita/artikel/detail/komitmen-pemerintah-melindungi-anak-di-ruang-digital

Kumparan. (2024). *180 Juta Orang Indonesia Main Medsos,
Rata-Rata Habiskan 3 Jam Sehari*.
https://kumparan.com/kumparannews/180-juta-orang-indonesia-main-medsos-rata-rata-habiskan-3-jam-sehari-275I28pbgaK

Twenge, J. M., & Campbell, W. K. (2019). Media Use Is Linked
to Lower Psychological Well-Being: Evidence from Three Datasets.
*Psychiatric Quarterly*, 90(2), 311–331.
https://doi.org/10.1007/s11126-019-09630-7