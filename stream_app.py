import os
import joblib
import requests
import pandas as pd
import streamlit as st

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Prediksi Risiko Kredit Amartha",
    page_icon="💳",
    layout="wide"
)

MODEL_ID = "1rVbvV7R-aHT8ScnuV0QRWwegwma-XZ5h"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "model.joblib"
FEATURES_PATH = "features.joblib"

# =========================
# STYLE (DIPERBARUI TANPA LOTTIE)
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f8f9fa; }
    
    .hero {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        margin-bottom: 2rem;
    }
    
    .card {
        padding: 1.5rem;
        border-radius: 15px;
        background: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh AI Model..."):
            r = requests.get(MODEL_URL, timeout=120)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_features():
    if not os.path.exists(FEATURES_PATH):
        st.error(f"File {FEATURES_PATH} tidak ditemukan!")
        st.stop()
    return joblib.load(FEATURES_PATH)

def group_features(feats):
    groups = {}
    for f in feats:
        key = f.split("_")[0] if "_" in f else "Lainnya"
        groups.setdefault(key, []).append(f)
    return {k: sorted(v) for k, v in sorted(groups.items(), key=lambda x: x[0].lower())}

# =========================
# MAIN APP
# =========================
# Sidebar simpel menggunakan standar Streamlit
with st.sidebar:
    st.title("📌 Menu Utama")
    st.info("Sistem ini memprediksi risiko kredit nasabah Amartha.")
    st.divider()
    st.markdown("### 🛠️ Status Sistem")
    st.success("Model: Ready")
    st.success("Features: Ready")

# Load Data
model = load_model()
features = load_features()
groups = group_features(features)
group_names = list(groups.keys())

# Header
st.markdown("""
<div class="hero">
    <h1 style="margin:0; color:white;">💳 Dashboard Prediksi Kredit</h1>
    <p style="opacity:0.9;">Sistem Cerdas Analisis Kelayakan Nasabah</p>
</div>
""", unsafe_allow_html=True)

col_form, col_res = st.columns([1.4, 1])

with col_form:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Form Data Nasabah")
    
    input_data = {}
    with st.form("form_prediksi"):
        c1, c2 = st.columns(2)
        mid = max(1, len(group_names) // 2)

        def render_form(group_list, container):
            for g in group_list:
                with container.expander(f"📍 {g}", expanded=True):
                    options = ["Pilih..."] + groups[g]
                    choice = st.selectbox(f"Pilih {g}", options, key=f"sb_{g}", label_visibility="collapsed")
                    for feat in groups[g]:
                        input_data[feat] = 1 if choice == feat else 0

        render_form(group_names[:mid], c1)
        render_form(group_names[mid:], c2)
        
        submitted = st.form_submit_button("🔍 Jalankan Analisis AI", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

with col_res:
    st.subheader("📊 Hasil Analisis")
    if not submitted:
        st.info("Silahkan lengkapi form di samping untuk memulai analisis.")
    
    if submitted:
        X = pd.DataFrame([input_data], columns=features)
        pred = int(model.predict(X)[0])
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if pred == 0:
            st.success("### ✅ HASIL: LANCAR")
            st.write("Nasabah dinilai aman untuk diberikan pinjaman.")
        else:
            st.error("### ❌ HASIL: MACET")
            st.write("Peringatan: Risiko gagal bayar terdeteksi tinggi.")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = proba.max()
            st.divider()
            st.metric(label="Tingkat Keyakinan AI", value=f"{confidence*100:.2f}%")
            st.progress(confidence)
        st.markdown('</div>', unsafe_allow_html=True)

st.caption("© 2026 Credit Risk Intelligence System | Amartha Dataset Analysis")