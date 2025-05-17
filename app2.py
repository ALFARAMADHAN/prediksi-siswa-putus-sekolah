# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. Baca data dengan separator yang benar
data = pd.read_csv('D:\Semester 5\MACHINE LEARNING\kelompok-ML\kelompok\jumlahsiswa.csv', sep=';')

# 2. Cek missing values
print("Missing Values: \n", data.isnull().sum())

# 3. Encoding kolom kategorikal
label_encoder_level = LabelEncoder()
label_encoder_daerah = LabelEncoder()

data['Level'] = label_encoder_level.fit_transform(data['Level'])  # Mengencode kolom Level
data['Daerah'] = label_encoder_daerah.fit_transform(data['Daerah'])  # Mengencode kolom Daerah

# 4. Pemilihan fitur dan target
X = data[['Level', 'Daerah']]  # Fitur umum
y_sd = data['SD']              # Target untuk SD
y_smp = data['SMP']            # Target untuk SMP
y_sma = data['SMA']            # Target untuk SMA
y_smk = data['SMK']            # Target untuk SMK
y_total = data['Total']        # Target total

# 5. Split data menjadi training dan testing
X_train, X_test, y_sd_train, y_sd_test = train_test_split(X, y_sd, test_size=0.2, random_state=42)
_, _, y_smp_train, y_smp_test = train_test_split(X, y_smp, test_size=0.2, random_state=42)
_, _, y_sma_train, y_sma_test = train_test_split(X, y_sma, test_size=0.2, random_state=42)
_, _, y_smk_train, y_smk_test = train_test_split(X, y_smk, test_size=0.2, random_state=42)
_, _, y_total_train, y_total_test = train_test_split(X, y_total, test_size=0.2, random_state=42)

# 6. Membangun model untuk setiap tingkat pendidikan
model_sd = LinearRegression().fit(X_train, y_sd_train)
model_smp = LinearRegression().fit(X_train, y_smp_train)
model_sma = LinearRegression().fit(X_train, y_sma_train)
model_smk = LinearRegression().fit(X_train, y_smk_train)
model_total = LinearRegression().fit(X_train, y_total_train)

# 7. Menampilkan Intersep dan Koefisien
print("\nIntersep dan Koefisien Model:")
print("Model SD - Intersep:", model_sd.intercept_, "| Koefisien:", model_sd.coef_)
print("Model SMP - Intersep:", model_smp.intercept_, "| Koefisien:", model_smp.coef_)
print("Model SMA - Intersep:", model_sma.intercept_, "| Koefisien:", model_sma.coef_)
print("Model SMK - Intersep:", model_smk.intercept_, "| Koefisien:", model_smk.coef_)
print("Model Total - Intersep:", model_total.intercept_, "| Koefisien:", model_total.coef_)

# 8. Evaluasi Model
# Prediksi menggunakan data pengujian
y_sd_pred = model_sd.predict(X_test)
y_smp_pred = model_smp.predict(X_test)
y_sma_pred = model_sma.predict(X_test)
y_smk_pred = model_smk.predict(X_test)
y_total_pred = model_total.predict(X_test)

print("\nEvaluasi Model:")
print("Model SD - R-squared:", r2_score(y_sd_test, y_sd_pred), "| MSE:", mean_squared_error(y_sd_test, y_sd_pred))
print("Model SMP - R-squared:", r2_score(y_smp_test, y_smp_pred), "| MSE:", mean_squared_error(y_smp_test, y_smp_pred))
print("Model SMA - R-squared:", r2_score(y_sma_test, y_sma_pred), "| MSE:", mean_squared_error(y_sma_test, y_sma_pred))
print("Model SMK - R-squared:", r2_score(y_smk_test, y_smk_pred), "| MSE:", mean_squared_error(y_smk_test, y_smk_pred))
print("Model Total - R-squared:", r2_score(y_total_test, y_total_pred), "| MSE:", mean_squared_error(y_total_test, y_total_pred))

# 9. Menentukan daerah dengan jumlah siswa putus sekolah tertinggi dan terendah
def daerah_tertinggi_terendah(model_sd, model_smp, model_sma, model_smk, model_total, label_encoder_daerah):
    semua_prediksi = []
    for idx in range(len(label_encoder_daerah.classes_)):
        daerah_encoded = idx
        data_input = pd.DataFrame({
            'Level': [label_encoder_level.transform(['Kabupaten/Kota'])[0]],
            'Daerah': [daerah_encoded]
        })
        prediksi_total = round(model_total.predict(data_input)[0])
        semua_prediksi.append((daerah_encoded, prediksi_total))
    
    # Menentukan daerah dengan jumlah tertinggi dan terendah
    daerah_tertinggi = max(semua_prediksi, key=lambda x: x[1])
    daerah_terendah = min(semua_prediksi, key=lambda x: x[1])
    
    nama_daerah_tertinggi = label_encoder_daerah.inverse_transform([daerah_tertinggi[0]])[0]
    nama_daerah_terendah = label_encoder_daerah.inverse_transform([daerah_terendah[0]])[0]
    
    return (nama_daerah_tertinggi, daerah_tertinggi[1], 
            nama_daerah_terendah, daerah_terendah[1])

# Menampilkan daerah dengan jumlah putus sekolah tertinggi dan terendah beserta tingkat pendidikannya
nama_daerah_tertinggi, jumlah_tertinggi, nama_daerah_terendah, jumlah_terendah = daerah_tertinggi_terendah(model_sd, model_smp, model_sma, model_smk, model_total, label_encoder_daerah)
print(f"Daerah dengan jumlah siswa putus sekolah tertinggi adalah {nama_daerah_tertinggi} dengan jumlah {jumlah_tertinggi}.")
print(f"Daerah dengan jumlah siswa putus sekolah terendah adalah {nama_daerah_terendah} dengan jumlah {jumlah_terendah}.")

# 10. Jika nama daerah valid, buat grafik perbandingan
nama_daerah = input("\nMasukkan nama daerah: ")

if nama_daerah in label_encoder_daerah.classes_:
    daerah_encoded = label_encoder_daerah.transform([nama_daerah])[0]
    
    # Buat data input untuk prediksi
    level_encoded = label_encoder_level.transform(['Kabupaten/Kota'])[0]  # Misal, semua prediksi untuk level Kabupaten/Kota
    data_input = pd.DataFrame({
        'Level': [level_encoded],
        'Daerah': [daerah_encoded]
    })

    # Prediksi jumlah siswa putus sekolah (dibulatkan)
    prediksi_sd = round(model_sd.predict(data_input)[0])
    prediksi_smp = round(model_smp.predict(data_input)[0])
    prediksi_sma = round(model_sma.predict(data_input)[0])
    prediksi_smk = round(model_smk.predict(data_input)[0])
    prediksi_total = round(model_total.predict(data_input)[0])

    # Tampilkan hasil prediksi
    print(f"\nPrediksi jumlah siswa putus sekolah untuk {nama_daerah} pada tahun 2025:")
    print(f"SD: {prediksi_sd}")
    print(f"SMP: {prediksi_smp}")
    print(f"SMA: {prediksi_sma}")
    print(f"SMK: {prediksi_smk}")
    print(f"Total keseluruhan: {prediksi_total}")
else:
    print("Nama daerah tidak valid. Pastikan nama sesuai dengan dataset.")
