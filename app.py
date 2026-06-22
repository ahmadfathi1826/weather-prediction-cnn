import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# Folder proyek
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path model
MODEL_PATH = os.path.join(BASE_DIR, "model_klasifikasi_cuaca.h5")

# Mapping label yang sudah kamu perbaiki
labels_map = {"0": "Cloudy", "1": "Rain", "2": "Shine", "3": "Sunrise"}

# Cek apakah file model benar-benar ada
if not os.path.exists(MODEL_PATH):
    st.error(f" File model tidak ditemukan di: {MODEL_PATH}")
    st.stop()


# Load model dengan cache
@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH, compile=False)


# Memuat model
try:
    model = load_my_model()
except Exception as e:
    st.error(f" Gagal memuat model: {e}")
    st.stop()

# Tampilan aplikasi
st.title(" Klasifikasi Cuaca")
st.write("Upload gambar untuk memprediksi kondisi cuaca")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Buka gambar
        img = Image.open(uploaded_file).convert("RGB")

        # Tampilkan gambar
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)

        # Preprocessing gambar
        img_resized = img.resize((150, 150))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Proses Prediksi
        with st.spinner("Sedang mengklasifikasi..."):
            pred = model.predict(img_array)

        # Ambil array probabilitas baris pertama
        probabilities = pred[0]

        # Ambil kelas dengan nilai tertinggi untuk hasil utama
        kelas_tertinggi = np.argmax(probabilities)
        confidence_tertinggi = probabilities[kelas_tertinggi] * 100
        label_prediksi = labels_map[str(kelas_tertinggi)]

        # 1. Tampilkan Hasil Utama
        st.success(
            f"**Prediksi Utama:** {label_prediksi} ({confidence_tertinggi:.2f}%)"
        )

        # 2. Tampilkan Detail Probabilitas untuk Semua Kelas
        st.write("---")
        st.subheader(" Detail Probabilitas Semua Kelas:")

        # Loop untuk menampilkan ke-4 persentase kelas secara rapi
        for i, prob in enumerate(probabilities):
            nama_kelas = labels_map[str(i)]
            persentase = prob * 100

            st.write(f"**{nama_kelas}**: {persentase:.2f}%")
            st.progress(float(prob))

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
