# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. Baca data dengan separator yang benar
data = pd.read_csv('jumlahsiswa.csv', sep=';')

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

# 9. Menerima input dari pengguna
nama_daerah = input("\nMasukkan nama daerah: ")

# Cek apakah nama daerah valid
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

# 10. Jika nama daerah valid, buat grafik perbandingan
if nama_daerah in label_encoder_daerah.classes_:
    # Data aktual tahun 2022/2023
    data_aktual = data[(data['Level'] == level_encoded) & (data['Daerah'] == daerah_encoded)].iloc[0]
    aktual_sd = data_aktual['SD']
    aktual_smp = data_aktual['SMP']
    aktual_sma = data_aktual['SMA']
    aktual_smk = data_aktual['SMK']
    aktual_total = data_aktual['Total']

    # Data prediksi tahun 2025
    prediksi = [prediksi_sd, prediksi_smp, prediksi_sma, prediksi_smk, prediksi_total]
    aktual = [aktual_sd, aktual_smp, aktual_sma, aktual_smk, aktual_total]

    # Nama kategori
    kategori = ['SD', 'SMP', 'SMA', 'SMK', 'Total']

    # Buat grafik
    x = np.arange(len(kategori))  # Label lokasi
    width = 0.35  # Lebar bar

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, aktual, width, label='Tahun 2022/2023', color='skyblue')
    bars2 = ax.bar(x + width/2, prediksi, width, label='Tahun 2025 (Prediksi)', color='orange')

    # Tambahkan label, judul, dan legenda
    ax.set_xlabel('Tingkat Pendidikan')
    ax.set_ylabel('Jumlah Siswa Putus Sekolah')
    ax.set_title(f'Perbandingan Jumlah Siswa Putus Sekolah untuk {nama_daerah}')
    ax.set_xticks(x)
    ax.set_xticklabels(kategori)
    ax.legend()

    # Tambahkan label nilai pada bar
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{round(bar.get_height())}', ha='center', va='bottom')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{round(bar.get_height())}', ha='center', va='bottom')

    # Tampilkan grafik
    plt.tight_layout()
    plt.show()
