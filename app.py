# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Teen Digital Wellbeing Predictor",
    layout="wide"
)

# ─────────────────────────────────────────────
# LOAD MODEL & SCALER
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load('models/classifier_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# ─────────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────────
COLOR_MAP = {
    'Healthy' : '#2ecc71',
    'Moderate': '#f39c12',
    'At Risk' : '#e74c3c'
}

RGBA_MAP = {
    'Healthy' : 'rgba(46, 204, 113, 0.15)',
    'Moderate': 'rgba(243, 156, 18, 0.15)',
    'At Risk' : 'rgba(231, 76, 60, 0.15)'
}

LABEL_MAP = {0: 'Healthy', 1: 'Moderate', 2: 'At Risk'}

REKOMENDASI = {
    0: {
        'deskripsi': (
            'Remaja ini menunjukkan pola perilaku digital yang sehat '
            'dan kesejahteraan mental yang baik. Pertahankan kebiasaan '
            'positif yang sudah terbentuk.'
        ),
        'saran': [
            'Pertahankan keseimbangan waktu penggunaan media sosial.',
            'Jaga rutinitas tidur yang teratur dan berkualitas.',
            'Terus aktif secara fisik minimal 30 menit per hari.',
            'Pertahankan interaksi sosial yang positif.',
            'Lakukan evaluasi berkala terhadap kebiasaan digital.',
        ]
    },
    1: {
        'deskripsi': (
            'Remaja ini menunjukkan beberapa tanda yang perlu '
            'diperhatikan terkait perilaku digital dan kesehatan '
            'mental. Diperlukan penyesuaian kebiasaan sebelum '
            'kondisi memburuk.'
        ),
        'saran': [
            'Batasi penggunaan media sosial maksimal 3 jam per hari.',
            'Hindari penggunaan gadget minimal 1 jam sebelum tidur.',
            'Tingkatkan aktivitas fisik setidaknya 30 menit per hari.',
            'Perbanyak interaksi sosial secara langsung (offline).',
            'Bicarakan perasaan dengan orang tua atau konselor sekolah.',
            'Terapkan jadwal penggunaan gadget yang terstruktur.',
        ]
    },
    2: {
        'deskripsi': (
            'Remaja ini menunjukkan indikator risiko tinggi terhadap '
            'gangguan kesehatan mental akibat perilaku digital yang '
            'tidak sehat. Diperlukan intervensi segera.'
        ),
        'saran': [
            'Segera konsultasikan dengan psikolog atau konselor profesional.',
            'Kurangi drastis penggunaan media sosial (maksimal 1-2 jam/hari).',
            'Terapkan digital detox secara bertahap dan terstruktur.',
            'Pastikan tidur 8-10 jam per malam dengan kualitas yang baik.',
            'Libatkan orang tua dalam mendampingi perubahan kebiasaan digital.',
            'Ikuti kegiatan ekstrakurikuler atau komunitas positif.',
            'Hindari penggunaan gadget di kamar tidur.',
        ]
    }
}

# ─────────────────────────────────────────────
# FUNGSI PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_input(data: dict) -> pd.DataFrame:
    social_map = {'low': 0, 'medium': 1, 'high': 2}
    sleep_map  = {'Poor': 0, 'Fair': 1, 'Good': 2}

    total_screen = (
        data['daily_social_media_hours'] + 
        data['screen_time_before_sleep']
    )
    sleep_eff = data['sleep_hours'] / (data['screen_time_before_sleep'] + 1)
    high_usage = int(data['daily_social_media_hours'] > 5)
    active     = int(data['physical_activity'] >= 1)

    row = {
        'age'                      : data['age'],
        'daily_social_media_hours' : data['daily_social_media_hours'],
        'sleep_hours'              : data['sleep_hours'],
        'screen_time_before_sleep' : data['screen_time_before_sleep'],
        'academic_performance'     : data['academic_performance'],
        'physical_activity'        : data['physical_activity'],
        'social_interaction_level' : social_map[data['social_interaction_level']],
        'stress_level'             : data['stress_level'],
        'anxiety_level'            : data['anxiety_level'],
        'addiction_level'          : data['addiction_level'],
        'total_screen_exposure'    : total_screen,
        'sleep_efficiency'         : sleep_eff,
        'high_social_media_usage'  : high_usage,
        'active_lifestyle'         : active,
        'sleep_quality_encoded'    : sleep_map[data['sleep_quality']],
        'gender_male'              : int(data['gender'] == 'Male'),
    }

    df_input  = pd.DataFrame([row])
    df_scaled = pd.DataFrame(
        scaler.transform(df_input),
        columns=df_input.columns
    )
    return df_scaled

# ─────────────────────────────────────────────
# FUNGSI VISUALISASI PROBABILITAS
# ─────────────────────────────────────────────
# Grafik probabilitas — update warna teks agar 
# terbaca di dark mode
def plot_probabilitas(proba: np.ndarray):
    kategori = ['Healthy', 'Moderate', 'At Risk']
    warna    = [COLOR_MAP[k] for k in kategori]
    nilai    = [p * 100 for p in proba]

    fig, ax = plt.subplots(figsize=(6, 3))

    # Background transparan agar ikut tema Streamlit
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    bars = ax.barh(kategori, nilai, color=warna, height=0.5)

    for bar, val in zip(bars, nilai):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%',
            va='center',
            fontsize=11,
            fontweight='bold',
            color='white'  # putih agar terbaca di dark mode
        )

    ax.set_xlim(0, 115)
    ax.set_xlabel('Probabilitas (%)', fontsize=10, color='white')
    ax.set_title('Probabilitas per Kategori', 
                 fontsize=12, pad=10, color='white')
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title('Teen Digital Wellbeing Predictor')
st.markdown(
    'Aplikasi prediksi status kesejahteraan digital remaja berdasarkan '
    'pola perilaku digital dan indikator kesehatan mental. '
    'Isi formulir di sidebar untuk mendapatkan hasil prediksi.'
)
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR — INPUT
# ─────────────────────────────────────────────
with st.sidebar:
    st.header('Data Remaja')
    st.caption('Isi seluruh data di bawah ini dengan lengkap.')

    st.subheader('Informasi Umum')
    age    = st.slider('Usia (tahun)', 13, 19, 16)
    gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'])

    st.subheader('Perilaku Digital')
    daily_social_media_hours = st.slider(
        'Durasi Media Sosial per Hari (jam)', 
        1.0, 8.0, 4.0, step=0.1
    )
    screen_time_before_sleep = st.slider(
        'Screen Time Sebelum Tidur (jam)', 
        0.5, 3.0, 1.5, step=0.1
    )
    addiction_level = st.slider(
        'Tingkat Kecanduan Media Sosial (1-10)', 
        1, 10, 5
    )

    st.subheader('Pola Tidur')
    sleep_hours   = st.slider(
        'Durasi Tidur per Malam (jam)', 
        4.0, 9.0, 7.0, step=0.1
    )
    sleep_quality = st.selectbox(
        'Kualitas Tidur', 
        ['Poor', 'Fair', 'Good']
    )

    st.subheader('Kesehatan & Akademik')
    academic_performance = st.slider(
        'Nilai Akademik (GPA 2.0 - 4.0)', 
        2.0, 4.0, 3.0, step=0.01
    )
    physical_activity = st.slider(
        'Aktivitas Fisik per Hari (jam)', 
        0.0, 2.0, 1.0, step=0.1
    )
    social_interaction_level = st.selectbox(
        'Tingkat Interaksi Sosial', 
        ['low', 'medium', 'high']
    )

    st.subheader('Kondisi Psikologis')
    stress_level  = st.slider('Tingkat Stres (1-10)', 1, 10, 5)
    anxiety_level = st.slider('Tingkat Kecemasan (1-10)', 1, 10, 5)

    st.divider()
    tombol = st.button(
        'Prediksi Sekarang', 
        type='primary', 
        use_container_width=True
    )

# ─────────────────────────────────────────────
# MAIN — HASIL PREDIKSI
# ─────────────────────────────────────────────
if tombol:
    input_data = {
        'age'                      : age,
        'gender'                   : gender,
        'daily_social_media_hours' : daily_social_media_hours,
        'screen_time_before_sleep' : screen_time_before_sleep,
        'sleep_hours'              : sleep_hours,
        'sleep_quality'            : sleep_quality,
        'academic_performance'     : academic_performance,
        'physical_activity'        : physical_activity,
        'social_interaction_level' : social_interaction_level,
        'stress_level'             : stress_level,
        'anxiety_level'            : anxiety_level,
        'addiction_level'          : addiction_level,
    }

    # Preprocessing & Prediksi
    X_input    = preprocess_input(input_data)
    pred_label = model.predict(X_input)[0]
    pred_proba = model.predict_proba(X_input)[0]
    pred_nama  = LABEL_MAP[pred_label]
    rec        = REKOMENDASI[pred_label]
    warna      = COLOR_MAP[pred_nama]

    # Layout dua kolom
    col_kiri, col_kanan = st.columns([1, 1], gap='large')

    with col_kiri:
        st.subheader('Hasil Prediksi')

        # Badge status hasil prediksi
        st.markdown(
            f"""
            <div style="
                background-color: {RGBA_MAP[pred_nama]};
                border-left: 5px solid {warna};
                padding: 16px 20px;
                border-radius: 8px;
                margin-bottom: 16px;
            ">
                <span style="
                    font-size: 22px;
                    font-weight: bold;
                    color: {warna};
                ">Status: {pred_nama}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(rec['deskripsi'])
        st.divider()

        # Grafik probabilitas
        fig = plot_probabilitas(pred_proba)
        st.pyplot(fig)

    with col_kanan:
        st.subheader('Rekomendasi')
        st.markdown(
            f'Berdasarkan hasil prediksi **{pred_nama}**, '
            'berikut langkah-langkah yang disarankan:'
        )

        # List rekomendasi
        for i, saran in enumerate(rec['saran'], start=1):
            st.markdown(
                f"""
                <div style="
                    background-color: rgba(255, 255, 255, 0.05);
                    border-radius: 8px;
                    padding: 10px 15px;
                    margin-bottom: 8px;
                    border-left: 3px solid {warna};
                ">
                    <span style="font-weight:bold; color:{warna};">
                        {i}.
                    </span> {saran}
                </div>
                """,
                unsafe_allow_html=True
            )

        # Detail fitur (expandable)
        st.divider()
        with st.expander('Lihat Ringkasan Data yang Diinput'):
            ringkasan = pd.DataFrame({
                'Variabel': [
                    'Usia', 'Jenis Kelamin', 'Durasi Media Sosial',
                    'Screen Time Sebelum Tidur', 'Durasi Tidur',
                    'Kualitas Tidur', 'Nilai Akademik',
                    'Aktivitas Fisik', 'Interaksi Sosial',
                    'Tingkat Stres', 'Tingkat Kecemasan',
                    'Tingkat Kecanduan'
                ],
                'Nilai': [
                    f'{age} tahun',
                    gender,
                    f'{daily_social_media_hours} jam/hari',
                    f'{screen_time_before_sleep} jam',
                    f'{sleep_hours} jam/malam',
                    sleep_quality,
                    f'{academic_performance:.2f}',
                    f'{physical_activity} jam/hari',
                    social_interaction_level,
                    f'{stress_level}/10',
                    f'{anxiety_level}/10',
                    f'{addiction_level}/10'
                ]
            })
            st.dataframe(
                ringkasan, 
                hide_index=True, 
                use_container_width=True
            )

else:
    # Tampilan default sebelum prediksi
    st.info(
        'Isi formulir di sidebar kiri, kemudian klik '
        'tombol "Prediksi Sekarang" untuk melihat hasil.'
    )

    st.subheader('Tentang Aplikasi')
    st.markdown(
        '''
        Aplikasi ini menggunakan model **Random Forest Classifier** 
        yang telah dilatih pada dataset perilaku digital dan 
        kesehatan mental remaja usia 13–19 tahun.

        Model memprediksi status kesejahteraan digital remaja ke 
        dalam tiga kategori:
        '''
    )

    col1, col2, col3 = st.columns(3)

    # Card kategori di halaman default
    with col1:
        st.markdown(
            f"""
            <div style="
                background-color: rgba(46, 204, 113, 0.15);
                border-left: 5px solid #2ecc71;
                padding: 16px;
                border-radius: 8px;
            ">
                <b style="color:#2ecc71; font-size:16px;">Healthy</b>
                <p style="margin-top:8px; font-size:14px;">
                Pola perilaku digital sehat dan kesejahteraan 
                mental yang baik.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="
                background-color: rgba(243, 156, 18, 0.15);
                border-left: 5px solid #f39c12;
                padding: 16px;
                border-radius: 8px;
            ">
                <b style="color:#f39c12; font-size:16px;">Moderate</b>
                <p style="margin-top:8px; font-size:14px;">
                Terdapat beberapa tanda yang perlu diperhatikan 
                dan diwaspadai.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div style="
                background-color: rgba(231, 76, 60, 0.15);
                border-left: 5px solid #e74c3c;
                padding: 16px;
                border-radius: 8px;
            ">
                <b style="color:#e74c3c; font-size:16px;">At Risk</b>
                <p style="margin-top:8px; font-size:14px;">
                Indikator risiko tinggi yang memerlukan 
                intervensi segera.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()
    st.caption(
        'Teen Digital Wellbeing Predictor — '
        'Proyek Machine Learning | CRISP-DM Framework'
    )
