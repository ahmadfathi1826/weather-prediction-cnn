# Klasifikasi Cuaca - Capstone Project

Aplikasi web sederhana untuk mengklasifikasikan kondisi cuaca dari gambar menggunakan model CNN (Convolutional Neural Network).

## Dataset
[Multiclass Weather Dataset](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset) (Kaggle)

## Cara Menjalankan

1. Clone repository ini
```bash
git clone https://github.com/ahmad-fathi18/weather-classification-cnn
cd weather-classification-cnn
```

2. Buat dan aktifkan virtual environment
```bash
# Membuat virtual environment
python -m venv venv

# Mengaktifkan (Windows)
venv\Scripts\activate

# Mengaktifkan (macOS / Linux)
source venv/bin/activate
```

3. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Jalankan aplikasi
```bash
streamlit run app.py
```

## Built With
- **Core:** Python 
- **Deep Learning:** TensorFlow / Keras
- **Web Framework:** Streamlit


## Author
* **Ahmad Fathi** - [GitHub Profile](https://github.com/ahmad-fathi18)
