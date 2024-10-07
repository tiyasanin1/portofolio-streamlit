# Hotel Reservation Cancellation Prediction

## Deskripsi
Proyek ini adalah aplikasi berbasis web yang dirancang untuk menganalisis dan memprediksi kemungkinan pembatalan reservasi hotel menggunakan data historis. Dengan menggunakan **Streamlit**, aplikasi ini memungkinkan pengguna untuk memasukkan parameter reservasi hotel dan mendapatkan prediksi mengenai apakah reservasi tersebut kemungkinan akan dibatalkan.

## Fitur
- **Input Pengguna**: Pengguna dapat memasukkan informasi tentang reservasi hotel seperti `lead time`, jenis hotel, dan jumlah tamu melalui antarmuka sidebar yang interaktif.
- **Visualisasi Data**:
  - **Distribusi Lead Time**: Histogram yang menunjukkan distribusi lead time berdasarkan status pembatalan.
  - **Segmen Pasar vs Pembatalan**: Grafik batang bertumpuk untuk menunjukkan hubungan antara segmen pasar dan status pembatalan.
- **Model Pembelajaran Mesin**: Menggunakan **Random Forest** untuk memprediksi apakah reservasi akan dibatalkan berdasarkan fitur yang dimasukkan.
- **Probabilitas Prediksi**: Menampilkan kemungkinan pembatalan untuk reservasi berdasarkan input pengguna.

## Dataset
Proyek ini menggunakan dataset `train.csv` yang berisi informasi tentang reservasi hotel, termasuk fitur-fitur yang relevan untuk analisis dan pemodelan.

## Teknologi yang Digunakan
- **Python**: Bahasa pemrograman yang digunakan.
- **Streamlit**: Framework untuk membangun aplikasi web interaktif.
- **Pandas**: Untuk manipulasi dan analisis data.
- **NumPy**: Untuk operasi matematika dan statistik.
- **Plotly**: Untuk visualisasi data interaktif.
- **Seaborn**: Untuk visualisasi statistik.
- **Scikit-learn**: Untuk membangun dan mengevaluasi model pembelajaran mesin.

## Cara Menggunakan
1. **Clone repository**:
   ```bash
   git clone https://github.com/tiyasanin1/portofolio-streamlit.git
