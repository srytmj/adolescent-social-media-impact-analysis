import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Teen & Young Adult Digital Wellbeing",
    layout="wide"
)

# --- 2. MEMUAT MODEL & SCALER ---
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('models/scaler.pkl')
        rf_model = joblib.load('models/classifier_model.pkl')
        return scaler, rf_model
    except Exception as e:
        st.error(f"Gagal memuat model. Error: {e}")
        return None, None

scaler, rf_model = load_models()

# --- 3. ANTARMUKA PENGGUNA (UI) ---
st.title("Screening Kesejahteraan Digital Remaja & Dewasa Muda")
st.markdown("""
Aplikasi ini memprediksi status **Kesejahteraan Digital** dan memberikan **rekomendasi psikologis yang dipersonalisasi** berdasarkan kombinasi unik dari pola perilaku harian Anda.
""")
st.divider()

if scaler and rf_model:
    st.subheader("Formulir Kebiasaan Harian")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### Profil Dasar & Sosial")
            age = st.slider("Umur (Tahun)", 13, 24, 16)
            gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
            social_interaction = st.selectbox("Tingkat Interaksi Sosial (Dunia Nyata)", ['Low', 'Medium', 'High'])
            physical_activity = st.slider("Aktivitas Fisik / Olahraga (Jam/Hari)", 0.0, 6.0, 1.0, step=0.5)

    with col2:
        with st.container(border=True):
            st.markdown("### Kebiasaan Digital & Tidur")
            daily_sosmed = st.slider("Total Durasi Main Sosmed (Jam/Hari)", 0.0, 18.0, 4.0, step=0.5)
            bedtime = st.selectbox("Jam Berapa Biasanya Mulai Tidur Malam?", 
                                   ["20:00", "21:00", "22:00", "23:00", "00:00", "01:00", "02:00", "03:00"])
            sleep_hours = st.slider("Berapa Jam Durasi Tidur Biasanya?", 3.0, 12.0, 7.0, step=0.5)
            screen_before_sleep = st.slider("Screen Time Sebelum Tidur (Jam)", 0.0, 5.0, 1.0, step=0.5)

    with col3:
        with st.container(border=True):
            st.markdown("### Kondisi Psikologis")
            st.caption("Self-Assessment")
            stress_level = st.slider("Tingkat Stres (1-10)", 1, 10, 5)
            anxiety_level = st.slider("Tingkat Kecemasan (1-10)", 1, 10, 5)
            addiction_level = st.slider("Tingkat Adiksi Gadget (1-10)", 1, 10, 5)

    # --- 4. TOMBOL PREDIKSI & LOGIKA LATAR BELAKANG ---
    st.divider()
    if st.button("Analisis Status & Dapatkan Rekomendasi", type="primary", use_container_width=True):
        with st.spinner("Sedang memproses data perilaku Anda secara spesifik..."):

            # A. Kalkulasi Kualitas Tidur
            bedtime_hour = int(bedtime.split(":")[0])
            if bedtime_hour < 12:  
                bedtime_hour += 24

            if sleep_hours >= 7 and screen_before_sleep <= 1.0 and bedtime_hour <= 23:
                calculated_sleep_quality = 2  # Good
            elif sleep_hours < 5.5 or screen_before_sleep >= 2.0 or bedtime_hour >= 24:
                calculated_sleep_quality = 0  # Poor
            else:
                calculated_sleep_quality = 1  # Fair

            soc_map = {'Low': 0, 'Medium': 1, 'High': 2}

            # Feature Engineering
            tot_screen = daily_sosmed + screen_before_sleep
            sleep_eff = sleep_hours / (screen_before_sleep + 1)
            high_sosmed = 1 if daily_sosmed > 5 else 0
            active_life = 1 if physical_activity >= 1 else 0

            # Menyiapkan Input untuk Model
            input_dict = {col: 0 for col in scaler.feature_names_in_}
            input_dict['academic_performance'] = 3.2 

            input_dict['age'] = age
            input_dict['daily_social_media_hours'] = daily_sosmed
            input_dict['sleep_hours'] = sleep_hours
            input_dict['screen_time_before_sleep'] = screen_before_sleep
            input_dict['physical_activity'] = physical_activity
            input_dict['stress_level'] = stress_level
            input_dict['anxiety_level'] = anxiety_level
            input_dict['addiction_level'] = addiction_level
            input_dict['social_interaction_level'] = soc_map[social_interaction]
            input_dict['sleep_quality_encoded'] = calculated_sleep_quality

            input_dict['total_screen_exposure'] = tot_screen
            input_dict['sleep_efficiency'] = sleep_eff
            input_dict['high_social_media_usage'] = high_sosmed
            input_dict['active_lifestyle'] = active_life

            gender_col = f"gender_{gender.lower()}"
            if gender_col in input_dict:
                input_dict[gender_col] = 1

            # --- BAGIAN PLATFORM DIHAPUS (Sesuai Permintaan) ---

            # Prediksi Model
            user_df = pd.DataFrame([input_dict])
            user_scaled = scaler.transform(user_df)

            pred_class = rf_model.predict(user_scaled)[0]
            pred_proba = rf_model.predict_proba(user_scaled)[0]

            status_labels = ["SEHAT (Healthy)", "SEDANG (Moderate)", "BERISIKO TINGGI (At Risk)"]
            sleep_labels = ["Buruk (Poor)", "Cukup (Fair)", "Baik (Good)"]

            # =========================================================================
            # --- MESIN REKOMENDASI DINAMIS (Menggunakan Tuple untuk Tipe Alert) ---
            # =========================================================================
            recommendations = []

            # 1. Pengecekan Kecemasan (Anxiety)
            if anxiety_level >= 7:
                if addiction_level >= 6 or daily_sosmed >= 5:
                    recommendations.append(("error", "**Kecemasan Terkait Digital:** Tingkat kecemasan Anda sangat tinggi bersamaan dengan tingginya paparan layar. Ini bisa jadi tanda overstimulasi atau FOMO (Fear of Missing Out). Sangat disarankan untuk mengurangi durasi sosmed secara bertahap."))
                else:
                    recommendations.append(("error", "**Indikasi Kecemasan Non-Digital:** Tingkat kecemasan Anda tinggi meskipun penggunaan gadget Anda masih wajar. Ini menandakan stresor dari luar (akademik, tekanan sosial, keluarga). Jangan ragu untuk bercerita ke konselor atau profesional."))

            # 2. Pengecekan Stres (Stress)
            if stress_level >= 7:
                if calculated_sleep_quality == 0:
                    recommendations.append(("error", "**Lingkaran Setan Stres & Kurang Tidur:** Stres Anda tinggi dan diperburuk oleh kualitas tidur yang buruk. Prioritaskan jam tidur. Matikan layar HP minimal 1 jam sebelum tidur agar pikiran bisa rileks."))
                else:
                    recommendations.append(("error", "**Manajemen Stres Darurat:** Anda mengalami tingkat stres yang mengkhawatirkan. Coba alihkan pikiran dengan melakukan aktivitas fisik di luar ruangan atau praktikkan pernapasan relaksasi setiap malam."))

            # 3. Pengecekan Adiksi (Addiction)
            if addiction_level >= 7:
                recommendations.append(("error", "**Peringatan Adiksi Gadget:** Anda menunjukkan indikasi ketergantungan smartphone yang tinggi. Mulailah gunakan fitur pembatas waktu aplikasi di pengaturan perangkat Anda."))

            # 4. Pengecekan Kebiasaan Malam (Screen Time)
            if screen_before_sleep >= 2.0:
                recommendations.append(("error", "**Bahaya Paparan Cahaya Biru:** Menatap layar 2 jam atau lebih sebelum tidur akan merusak produksi hormon melatonin. Kebiasaan ini dapat memicu insomnia berkepanjangan."))

            # 5. Pengecekan Aktivitas Fisik
            if physical_activity < 1.0:
                recommendations.append(("warning", "**Tubuh Kurang Gerak:** Aktivitas fisik Anda di bawah 1 jam. Olahraga ringan (seperti jalan santai) adalah cara paling alami dan ampuh untuk membuang hormon stres dari tubuh."))

            # 6. Pengecekan Kehidupan Sosial
            if social_interaction == 'Low':
                recommendations.append(("warning", "**Waspada Isolasi Sosial:** Interaksi di dunia nyata Anda sangat rendah. Cobalah untuk sekadar mengobrol langsung dengan keluarga di rumah atau teman untuk menjaga keseimbangan sosial Anda."))

            # 7. Fallback (Jika sehat)
            if len(recommendations) == 0:
                if stress_level <= 4 and anxiety_level <= 4 and addiction_level <= 4:
                    recommendations.append(("success", "**Pertahankan Gaya Hidup Ini!** Pola tidur, durasi penggunaan media sosial, dan indikator stres/kecemasan Anda berada dalam harmoni yang sangat baik. Lanjutkan rutinitas positif Anda."))
                else:
                    recommendations.append(("warning", "**Perhatian Ringan:** Meski kebiasaan harian Anda secara umum cukup baik, pastikan tingkat stres atau adiksi gadget tidak dibiarkan meningkat secara perlahan."))
            # =========================================================================

            # --- TAMPILAN OUTPUT (TETAP SAMA) ---
            st.subheader("Hasil Analisis Terkini")
            res_col1, res_col2 = st.columns([1, 1.5])

            with res_col1:
                if pred_class == 0:
                    st.success(f"### {status_labels[0]}")
                elif pred_class == 1:
                    st.warning(f"### {status_labels[1]}")
                else:
                    st.error(f"### {status_labels[2]}")

                st.write(f"**Kualitas Tidur (Estimasi):** {sleep_labels[calculated_sleep_quality]}")
                st.write("**Probabilitas Prediksi:**")
                st.write(f"- Sehat: {pred_proba[0]*100:.1f}%")
                st.write(f"- Sedang: {pred_proba[1]*100:.1f}%")
                st.write(f"- Berisiko: {pred_proba[2]*100:.1f}%")

            with res_col2:
                st.markdown(f"**Rekomendasi Khusus Untuk Anda ({len(recommendations)} Poin Penting):**")

                # Render rekomendasi berdasarkan tipe tuple
                for alert_type, text in recommendations:
                    if alert_type == "error":
                        st.error(text)
                    elif alert_type == "success":
                        st.success(text)
                    else:
                        st.warning(text)
