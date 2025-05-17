from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Inisialisasi Flask
app = Flask(__name__)

# Load dataset
data = pd.read_csv('D:\Semester 5\MACHINE LEARNING\kelompok-ML\kelompok\jumlahsiswa.csv', sep=';')

# Menghapus baris dengan nilai NaN
data = data.dropna()

# Preprocessing
label_encoder_level = LabelEncoder()
label_encoder_daerah = LabelEncoder()

# Mengonversi data kategorikal menjadi numerik
data['Level'] = label_encoder_level.fit_transform(data['Level'])
data['Daerah'] = label_encoder_daerah.fit_transform(data['Daerah'])

# Pemisahan fitur dan target
X = data[['Level', 'Daerah']]
y_sd = data['SD']
y_smp = data['SMP']
y_sma = data['SMA']
y_smk = data['SMK']
y_total = data['Total']

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_sd_train, y_sd_test = train_test_split(X, y_sd, test_size=0.2, random_state=42)
_, _, y_smp_train, y_smp_test = train_test_split(X, y_smp, test_size=0.2, random_state=42)
_, _, y_sma_train, y_sma_test = train_test_split(X, y_sma, test_size=0.2, random_state=42)
_, _, y_smk_train, y_smk_test = train_test_split(X, y_smk, test_size=0.2, random_state=42)
_, _, y_total_train, y_total_test = train_test_split(X, y_total, test_size=0.2, random_state=42)

# Melatih model
model_sd = LinearRegression().fit(X_train, y_sd_train)
model_smp = LinearRegression().fit(X_train, y_smp_train)
model_sma = LinearRegression().fit(X_train, y_sma_train)
model_smk = LinearRegression().fit(X_train, y_smk_train)
model_total = LinearRegression().fit(X_train, y_total_train)

# Halaman utama
@app.route('/')
def home():
    daerah_list = label_encoder_daerah.classes_
    return render_template('index.html', daerah_list=daerah_list)

# Prediksi jumlah siswa berdasarkan daerah
@app.route('/predict', methods=['POST'])
def predict():
    daerah = request.form['daerah']

    if daerah not in label_encoder_daerah.classes_:
        return jsonify({'error': f'Nama daerah "{daerah}" tidak valid. Pastikan nama sesuai dataset.'})

    # Encode input daerah
    daerah_encoded = label_encoder_daerah.transform([daerah])[0]
    level_encoded = label_encoder_level.transform(['Kabupaten/Kota'])[0]  # Default Level
    
    data_input = pd.DataFrame({'Level': [level_encoded], 'Daerah': [daerah_encoded]})

    # Prediksi
    prediksi_sd = round(model_sd.predict(data_input)[0])
    prediksi_smp = round(model_smp.predict(data_input)[0])
    prediksi_sma = round(model_sma.predict(data_input)[0])
    prediksi_smk = round(model_smk.predict(data_input)[0])
    prediksi_total = round(model_total.predict(data_input)[0])

    hasil = {
        'SD': prediksi_sd,
        'SMP': prediksi_smp,
        'SMA': prediksi_sma,
        'SMK': prediksi_smk,
        'Total': prediksi_total
    }
    return jsonify(hasil)

if __name__ == '__main__':
    app.run(debug=True)
