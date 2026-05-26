# Screening Kesejahteraan Digital Remaja dan Dewasa Muda

Proyek ini dibuat untuk menganalisis dan memprediksi kondisi kesejahteraan digital pada remaja dan dewasa muda (usia 13–24 tahun). Output dari model berupa tiga kategori: Sehat, Sedang, dan Berisiko Tinggi.

Prediksi dilakukan menggunakan algoritma Random Forest dengan mempertimbangkan beberapa faktor, seperti kebiasaan penggunaan media sosial, pola tidur, aktivitas fisik, serta kondisi psikologis seperti stres, kecemasan, dan tingkat ketergantungan terhadap gadget.

Selain prediksi, aplikasi ini juga menyediakan rekomendasi sederhana yang disesuaikan dengan kondisi pengguna, berdasarkan aturan logika yang dibuat dari kombinasi input yang diberikan.

## Struktur Direktori

```

customer-response-prediction/
├── data/          # Dataset mentah dan hasil preprocessing
├── models/        # File model dan scaler (.pkl)
├── notebooks/     # Notebook proses analisis
├── app.py         # Aplikasi utama berbasis Streamlit
└── requirements.txt

````

## Alur Pengerjaan

1. Data Understanding  
2. Data Cleaning & Feature Engineering  
3. Exploratory Data Analysis (EDA)  
4. Data Preprocessing  
5. Modeling (Random Forest)  
6. Evaluation  
7. Deployment ke Streamlit  

## Fitur Aplikasi

- **Form Input Interaktif**  
  Pengguna mengisi data seperti umur, aktivitas fisik, penggunaan media sosial, pola tidur, serta kondisi psikologis.

- **Perhitungan Kualitas Tidur Otomatis**  
  Sistem mengkategorikan kualitas tidur berdasarkan jam tidur, durasi, dan penggunaan layar sebelum tidur.

- **Prediksi Status Kesejahteraan Digital**  
  Model memberikan hasil klasifikasi beserta probabilitasnya.

- **Rekomendasi Sederhana**  
  Sistem memberikan saran berdasarkan kondisi pengguna, misalnya terkait stres tinggi, kurang tidur, atau penggunaan gadget berlebih.

## Cara Menjalankan

1. Pastikan Python sudah terpasang (disarankan versi 3.9 ke atas).
2. Masuk ke folder proyek melalui terminal.
3. Install dependency:

```bash
pip install -r requirements.txt
````

4. Jalankan aplikasi:

```bash
streamlit run app.py
```

5. Aplikasi akan terbuka di browser. Jika tidak, buka manual di:
   [http://localhost:8501](http://localhost:8501)

---
Dataset: https://www.kaggle.com/datasets/abdulmaliklodhra/social-media-addiction-and-mental-health-dataset/data