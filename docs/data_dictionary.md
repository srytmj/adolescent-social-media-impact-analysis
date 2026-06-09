[Kembali ke README](../../README.md)

# Data Dictionary — Teen Mental Health Dataset

Sumber: [Kaggle — argonnxx/teen-mental-health](https://www.kaggle.com/datasets/argonnxx/teen-mental-health)  
Total data: ±1.200 baris | Rentang usia: 13–19 tahun | Jenis data: Sintetis

---

## Kolom Original

| No | Nama Kolom | Tipe | Rentang / Kategori | Deskripsi |
|---|---|---|---|---|
| 1 | `age` | Integer | 13–19 | Usia responden dalam tahun |
| 2 | `gender` | Kategorikal | male, female | Jenis kelamin responden |
| 3 | `daily_social_media_hours` | Float | 1.0–8.0+ | Rata-rata jam penggunaan media sosial per hari |
| 4 | `platform_usage` | Kategorikal | TikTok, Instagram, YouTube, Facebook, All Platforms | Platform media sosial yang paling sering digunakan |
| 5 | `sleep_hours` | Float | 4.0–9.0 | Rata-rata jam tidur per malam |
| 6 | `screen_time_before_sleep` | Float | 0.5–3.0 | Durasi penggunaan layar sebelum tidur (jam) |
| 7 | `academic_performance` | Float | 2.00–4.00 | Skor performa akademik (ekuivalen GPA) |
| 8 | `physical_activity` | Float | 0–2.0 | Durasi aktivitas fisik per hari (jam) |
| 9 | `social_interaction_level` | Kategorikal | low, medium, high | Tingkat interaksi sosial di dunia nyata |
| 10 | `stress_level` | Integer | 1–10 | Tingkat stres (1=rendah, 10=sangat tinggi) — self-report |
| 11 | `anxiety_level` | Integer | 1–10 | Tingkat kecemasan (1=rendah, 10=sangat tinggi) — self-report |
| 12 | `addiction_level` | Integer | 1–10 | Tingkat perilaku adiktif terhadap gadget (1=rendah, 10=sangat tinggi) — self-report |
| 13 | `depression_label` | Integer | 0, 1 | Indikator depresi (0=tidak, 1=ya) — kelas sangat imbalanced, mayoritas 0 |
| 14 | `mental_health_risk_score` | Integer | 3–30 | Skor risiko kesehatan mental agregat |
| 15 | `sleep_quality` | Kategorikal | Poor, Fair, Good | Kategori kualitas tidur |
| 16 | `digital_wellbeing_flag` | Kategorikal | At Risk, Moderate, Healthy | **Target variabel** — status kesejahteraan digital |

---

## Kolom Hasil Feature Engineering (NB2)

| No | Nama Kolom | Formula | Deskripsi |
|---|---|---|---|
| 1 | `total_screen_exposure` | `daily_social_media_hours + screen_time_before_sleep` | Total paparan layar per hari |
| 2 | `sleep_efficiency` | `sleep_hours / (screen_time_before_sleep + 1)` | Rasio efisiensi tidur terhadap paparan layar malam |
| 3 | `high_social_media_usage` | `1 if daily_social_media_hours > 5 else 0` | Flag penggunaan media sosial tinggi |
| 4 | `active_lifestyle` | `1 if physical_activity >= 1 else 0` | Flag gaya hidup aktif secara fisik |
| 5 | `risk_category` | Berbasis kuartil: Q1≤13=Low, 14–20=Medium, >20=High | Kategori risiko dari `mental_health_risk_score` |

---

## Kolom yang Di-drop

| Nama Kolom | Alasan |
|---|---|
| `platform_usage` | Tidak deterministik terhadap target; terlalu banyak kategori nominal |
| `depression_label` | Data leakage — berkorelasi langsung dengan target `digital_wellbeing_flag` |
| `mental_health_risk_score` | Data leakage — merupakan skor agregat pembentuk target |
| `risk_category` | Derived dari `mental_health_risk_score`, berpotensi leakage |
| `sleep_quality` | Di-drop setelah encoding di NB4 (sudah digantikan fitur numerik tidur) |

---

## Catatan

- Dataset bersifat **sintetis** — relasi antar variabel kemungkinan direkayasa untuk keperluan pembelajaran, bukan kausalitas dunia nyata.
- Kolom self-report (`stress_level`, `anxiety_level`, `addiction_level`) rentan terhadap **response bias**.
- Tidak mengandung PII (Personally Identifiable Information).
- Data bersifat **cross-sectional** (snapshot), bukan longitudinal.
