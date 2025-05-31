# Laporan Proyek Machine Learning - Arif Maulana Insaf

## analysis sentimen
Dalam industri telekomunikasi Indonesia yang sangat kompetitif, pelanggan semakin bergantung pada ulasan digital untuk mengevaluasi kualitas layanan. Data dari Badan Regulasi Telekomunikasi Indonesia (BRTI, 2023) menunjukkan bahwa keluhan pelanggan operator seluler mencapai 5.782 laporan per bulan, dengan 43% terkait masalah kuota dan jaringan. Namun, sebagian besar perusahaan masih mengandalkan analisis manual yang lambat dan subjektif.

Proyek ini mengembangkan solusi klasifikasi sentimen otomatis berbasis deep learning untuk:
- Mengkategorikan ulasan pelanggan IM3 menjadi positif, netral, atau negatif
- Mengidentifikasi topik keluhan dominan (kuota, jaringan, layanan pelanggan)
- Memberikan insight real-time untuk perbaikan layanan

referensi 
Dang, N. C., Moreno-García, M. N., & De la Prieta, F. (2020).
Sentiment analysis based on deep learning: A comparative study.
[https://www.mdpi.com/2079-9292/9/3/483](https://www.mdpi.com/2079-9292/9/3/483)

# Business Understanding
Bagian laporan ini mencakup : 
### Problem Statements
- Bagaimana cara membangun model klasifikasi sentimen yang akurat untuk ulasan pelanggan IM3 dalam bahasa Indonesia, mampu mengkategorikan sentimen menjadi positif, netral, dan negatif?
- Bagaimana model dapat secara efektif mengidentifikasi topik keluhan dominan (kuota, jaringan, layanan pelanggan) dari ulasan yang diklasifikasikan sebagai negatif atau netral?
### Goals
- saya Mengembangkan model klasifikasi sentimen deep learning yang mampu mengkategorikan ulasan pelanggan IM3 menjadi positif, netral, dan negatif
- Memberikan kemampuan untuk secara implisit atau eksplisit mengidentifikasi topik keluhan dominan (kuota, jaringan, layanan pelanggan) dari ulasan, yang dapat diukur melalui analisis distribusi topik dalam sentimen negatif/netral.
### solution 
Berdasarkan hasil evaluasi yang diberikan untuk kedua algoritma, yaitu LSTM dan CNN, dapat disimpulkan model terbaik sebagai solusi.

Modeling
Pemilihan Model Terbaik sebagai Solusi
Dalam proyek ini, kami mengajukan dua algoritma deep learning yang berbeda untuk klasifikasi sentimen ulasan pelanggan: Long Short-Term Memory (LSTM) dan Convolutional Neural Network (CNN). Setelah melatih dan mengevaluasi kedua model, kami memilih model CNN sebagai solusi terbaik.

Perbandingan Hasil Model:

1. Model LSTM:

Akurasi: 0.5265 (sekitar 53%)
Classification Report:

|katergori| precision | recall | f1-score |
|--------|-------|-----------------|-------|
|negatif:| 0.53| 1.00| 0.69|
|netral:|  0.00| 0.00| 0.00|
|positif:| 0.00|  0.00|  0.00|

Macro Avg F1-Score: 0.23

Weighted Avg F1-Score: 0.36

Dari hasil ini, terlihat bahwa model LSTM memiliki masalah serius dalam mengklasifikasikan kelas 'netral' dan 'positif', di mana precision, recall, dan F1-score untuk kedua kelas tersebut adalah 0.00. Ini mengindikasikan bahwa model LSTM cenderung memprediksi semua ulasan ke dalam kelas 'negatif' saja, sehingga akurasi keseluruhan menjadi menyesatkan.

2. Model CNN:

Akurasi: 0.7990 (sekitar 80%)
Classification Report:

|kategori | precission | recall | f1-score |
|---------|---------|---------|----------|
|negatif: |0.87| 0.83| 0.85|
|netral: | 0.15 | 0.10 |0.12|
|positif: |0.79| 0.87| 0.83|

Macro Avg F1-Score: 0.60

Weighted Avg F1-Score: 0.79

Mengapa Model CNN Dipilih sebagai Model Terbaik:

Model CNN dipilih sebagai solusi terbaik berdasarkan alasan-alasan berikut:

Akurasi Keseluruhan yang Jauh Lebih Tinggi: Model CNN mencapai akurasi sebesar 80%, yang jauh lebih tinggi dibandingkan dengan akurasi model LSTM yang hanya 53%. Akurasi yang lebih tinggi menunjukkan bahwa model CNN secara keseluruhan lebih sering membuat prediksi yang benar.
Mampu Mengklasifikasikan Semua Kelas: Berbeda dengan model LSTM yang gagal total dalam memprediksi kelas 'netral' dan 'positif', model CNN menunjukkan kemampuan untuk memprediksi ketiga kelas sentimen ('negatif', 'netral', 'positif'), meskipun performa pada kelas 'netral' masih rendah. Kemampuan untuk membedakan antara ketiga sentimen ini sangat krusial untuk tujuan proyek, yaitu mendapatkan insight yang beragam.
Performa yang Baik pada Kelas Mayoritas: Model CNN menunjukkan precision, recall, dan F1-score yang solid (di atas 0.79) untuk kelas 'negatif' dan 'positif'. Ini sangat penting karena kedua kelas ini kemungkinan merupakan mayoritas ulasan yang akan ditangani.
Keseimbangan Metrik: Meskipun kelas 'netral' masih menjadi tantangan bagi CNN (dengan F1-score 0.12), kinerja yang kuat pada kelas 'negatif' dan 'positif' serta akurasi keseluruhan yang tinggi menjadikannya pilihan yang lebih unggul. F1-score weighted average CNN sebesar 0.79 juga jauh lebih baik dibandingkan LSTM (0.36), menunjukkan kinerja yang lebih seimbang secara keseluruhan.

# Data understanding
Dataset yang digunakan dalam proyek ini ialah im3 yang saya ambil dari kaggle 
#### sumber datasat
tautan : [myIM3_Beli_Pulsa__Cek_Kuota.csv](https://www.kaggle.com/datasets/ucupsedaya/10k-myim3-app-reviews)
informasi dataset : 
- jumlah data : 10.000 data
- jumlah column : 5 column
- ukuran dataset : (10.000, 5)
- kondisi data : masih terdapat column yang null/missing value yaitu apa appVersion

## variabel - variabel dataset
- userName: Nama pengguna atau pemberi ulasan pada myIM3. Identifikasi unik atau pseudonim yang menunjukkan siapa yang memberikan ulasan.
- content: Isi ulasan yang diberikan oleh pengguna. Kolom ini berisi teks dari ulasan yang mencakup pengalaman atau pendapat pengguna terkait aplikasi myIM3.
- score: Skor yang diberikan oleh pengguna terhadap aplikasi myIM3. Nilai skor ini umumnya berkisar antara 1 hingga 5, di mana 1 mungkin menunjukkan ketidakpuasan dan 5 menunjukkan kepuasan tinggi.
- at: Tanggal ulasan diberikan oleh pengguna. Ini mencatat kapan ulasan tersebut diunggah atau diberikan.
- appVersion: Versi aplikasi myIM3 yang digunakan oleh pengguna saat memberikan ulasan. Informasi ini membantu dalam memahami konteks pembaruan aplikasi yang mungkin mempengaruhi pengalaman pengguna.

## deskripsi variabel 

| # | column | non-null count | Dtype |
|---|------|-------|-----|
| 1 | userName | 10000 non-null | object |
| 2 | cotent | 10000 non-null | object |
| 3 | score | 10000 non-null | int | 
| 4 | at | 10000 non-null | object | 
| 4 | appVersion | 8026 non-null | object |

dari dataset diatas ada beberapa fitur kategorikal dan numerikal:
- terdapat 4 fitur kategorikal yaitu `userName`, `content`, `at`, dan `appVersion`
- numerikal hanya 1 fitur yaitu `score`

dan dari data di atas 3 fitur yang di hapus yatu 
- userName = nama dihapus karena saya tidak di butuhkan untuk analysis sentimen
- at = at dihapus kerana untuk analysis sentimen tidak membutuhkan tanggal
- appVersion = appVersion tidak dibutuhkan karena untuk analysis sentimen tidak membutuhkan versi dari aplikasi

## exploratory data analysis (EDA) 
pengecekan missing value 

| # | columns | value |
|---|---------|-------|
| 1 | userName | 0 |
| 2 | content |  0 |
| 3 | score | 0 |
| 4 | at | 0 |
| 5 | appVersion | 1974 |

pengecekan duplicate 

```
np.int64(0)
```
## eda 

![image](https://github.com/user-attachments/assets/894cfbdd-e315-4e19-bf75-a6a5b43317eb)

 Berdasarkan grafik Distribusi Panjang Teks Review, berikut interpretasi hasilnya:

1. Karakteristik Distribusi
Range Panjang Teks:
- Mayoritas ulasan memiliki panjang 0–100 karakter (puncak tertinggi).
- Terdapat beberapa ulasan panjang (>300 karakter), tetapi sangat jarang.
Distribusi:
- Right-skewed (ekor memanjang ke kanan), menunjukkan sebagian besar teks pendek dengan sedikit outlier panjang.


# data prepration 

| # | tenik yang digunakan |  alasan |
|---|---------------------|--------|
| 1 | drop_columns() | saya menghapus beberapa columns yang menurut saya tidak penting dan tidak akan saya gunakan untuk analysis sentimen |
| 2 | membuat function preprocess | saya membuat fungtion tersebut untuk di terapkan pada column `content ` yang mana berisikan<br>1. konversi teks ke huruf kecil<br> 2. penghapusan url, mention, dan hashtaq<br>3. menghapus karakter non-alfabet<br> 4. normalisasi data<br> 5. tokenisasi dan stopword removal<br> 6. penggabungan kembali kek tokens |
| 3 | membuat tokenisasi dengan tokenize | membuat tokenisasi pada column `clean_text` yang akan direpresentasikan menjadi nilai numerik |
| 4 | pad_sequence | untuk menyamakan deretan angka hasil tokenisasi agar bisa di proses oleh model dalam batch |
| 5 | membuat function sentiment | columns score  agar bisa di kategorikan sebagai positf, negatif, dan netral | 
| 6 | labelencoder | columns sentiment yang telah di dapatkan pada function seblumnya akan di labelkan mengunakan labelencoder |
| 7 | splitting data | splitting data hasil dari padde sebagai x dan column label sebagian y |


# modeling 

pada model saya menggunakan model lstm yang terdiri dari 7 layer 

```python
model = Sequential([
    Embedding(vocab_size, 128),          # Layer 1
    LSTM(128, return_sequences=True),    # Layer 2
    Dropout(0.5),                         # Layer 3
    LSTM(64),                             # Layer 4
    Dropout(0.5),                         # Layer 5
    Dense(64, activation='relu'),         # Layer 6
    Dense(3, activation='softmax')        # Layer 7
])
```
| Layer                              | Fungsi                                                                             |
| ---------------------------------- | ---------------------------------------------------------------------------------- |
| `Embedding(vocab_size, 128)`       | Mengubah token kata menjadi vektor densitas (word embeddings) berdimensi 128       |
| `LSTM(128, return_sequences=True)` | LSTM pertama dengan 128 unit, mengembalikan seluruh urutan (untuk LSTM berikutnya) |
| `Dropout(0.5)`                     | Mencegah overfitting dengan meng-nol-kan 50% neuron secara acak                    |
| `LSTM(64)`                         | LSTM kedua, menangkap pola temporal secara lebih abstrak                           |
| `Dropout(0.5)`                     | Dropout kedua untuk regularisasi tambahan                                          |
| `Dense(64, relu)`                  | Layer dense untuk abstraksi lanjutan sebelum output                                |
| `Dense(3, softmax)`                | Layer output, memetakan ke 3 kelas: **negatif**, **netral**, **positif**           |

dan pada proses  pelatihannya saya menggunakan callback

```python
early_stop = EarlyStopping(monitor='val_loss', patience=5)
```
yang mana jika `val_loss` tidak membaik dalam 3 epoch, maka proses train akan di hentikan 


## lstm 
### kelebihan model lstm
- mampu menangkap urutan kata dan konteks antar kata
- lstm efektif untuk menangani data teks panjang
- tidak memerlukan fitur enggineering manual seperti tf-idf


### kekurangan model lstm 
- membutuhkan data yang besar
- sulit di interprestasikan
- rentan overfitting


## cnn
### kelebihan model cnn
- robust terhadapan panjang input
- tidak rentan terhadap vanishing gradient
- menangkap fitur lokal

###
- tidak memahami konteks seqquential
- sensitif terhadap ukuran kernel
- kurang umum untuk pemahaman konteks

# evaluasi 

TP (True Positive): Prediksi benar kelas positif.
TN (True Negative): Prediksi benar kelas negatif.
FP (False Positive): Prediksi salah kelas positif (seharusnya negatif).
FN (False Negative): Prediksi salah kelas negatif (seharusnya positif).


###  matrik yang di gunakan 
  1. Accuracy
   Penjelasan: Proporsi prediksi benar secara keseluruhan.
#### formula
```
(TP + TN) / (TP + TN + FP + FN)
```
#### cara kerja:
  - Model melakukan prediksi untuk setiap ulasan di dataset uji.
  - Kemudian, dihitung berapa banyak prediksi tersebut yang cocok dengan label sebenarnya (misalnya, ulasan positif diprediksi positif, ulasan netral diprediksi netral, ulasan negatif diprediksi negatif).
  - Jumlah prediksi yang benar ini (TP untuk semua kelas yang benar diprediksi, dan TN untuk semua kelas yang benar diprediksi tidak termasuk kelas target) dibagi dengan total jumlah ulasan yang dievaluasi.

  2. precision
Penjelasan: Ketepatan prediksi positif (berapa banyak prediksi kelas X yang benar benar kelas X).
#### formula
```
TP / (TP + FP)
```
#### cara kerja: 
- Fokus pada satu kelas tertentu, misalnya sentimen 'negatif'.
- Model melihat semua ulasan yang diprediksi sebagai 'negatif'.
- Kemudian, dihitung berapa banyak dari prediksi 'negatif' tersebut yang benar-benar 'negatif' (TP_negatif).
- Jumlah TP_negatif ini dibagi dengan total semua yang diprediksi 'negatif' (yaitu, ulasan yang benar-benar negatif dan diprediksi negatif (TP_negatif) ditambah ulasan yang sebenarnya bukan negatif tetapi salah diprediksi negatif (FP_negatif)).

  
  3. recall
penjelasaan: Kemampuan model menemukan semua instance kelas X.
#### formula
```
TP / (TP + FN)
```
#### cara kerja 
- Fokus pada satu kelas tertentu, misalnya sentimen 'negatif'.
- Model melihat semua ulasan yang sebenarnya berlabel 'negatif' di dataset uji.
- Kemudian, dihitung berapa banyak dari ulasan 'negatif' yang sebenarnya tersebut yang berhasil diprediksi dengan benar sebagai 'negatif' (TP_negatif).
- Jumlah TP_negatif ini dibagi dengan total semua yang sebenarnya 'negatif' (yaitu, ulasan yang benar-benar negatif dan diprediksi negatif (TP_negatif) ditambah ulasan yang benar-benar negatif tetapi salah diprediksi bukan negatif (FN_negatif)).
  
  4. f1-score
penjelasan:  Rata-rata harmonik precision dan recall (baik untuk data tidak seimbang).
#### formula 
```
 2 * (Precision * Recall) / (Precision + Recall)
```
#### cara kerja:
- Setelah menghitung Presisi dan Recall untuk setiap kelas, F1-Score dihitung menggunakan formula rata-rata harmonik.
- F1-Score memberikan gambaran yang lebih seimbang tentang kinerja model dibandingkan hanya menggunakan Presisi atau Recall secara terpisah. Jika salah satu nilai (Presisi atau Recall) sangat rendah, F1-Score juga akan rendah, menunjukkan bahwa model tidak berfungsi dengan baik secara keseluruhan untuk kelas tersebut.

  
  5. macro avg
  penjelasan: Rata-rata metrik tiap kelas (sama pentingnya untuk semua kelas).
  #### cara kerja:
  - Hitung Presisi, Recall, dan F1-Score untuk setiap kelas ('negatif', 'netral', 'positif').
  - Jumlahkan nilai-nilai Presisi dari ketiga kelas, lalu bagi dengan jumlah kelas (3). Lakukan hal yang sama untuk Recall dan F1-Score.  
  
  6. weighted avg
  penjelasan: Rata-rata metrik yang dibobotkan berdasarkan jumlah sampel tiap kelas.
 #### cara kerja:
 - Hitung Presisi, Recall, dan F1-Score untuk setiap kelas.
 - Untuk setiap metrik, kalikan nilai metrik kelas dengan jumlah sampel (support) kelas tersebut, lalu jumlahkan hasilnya untuk semua kelas, dan bagi dengan total jumlah sampel dari semua kelas.

## analisis hasil 

### hasil dari lstm 

Akurasi: 0.5265 (sekitar 53%)
Classification Report:

|katergori| precision | recall | f1-score |
|--------|-------|-----------------|-------|
|negatif:| 0.53| 1.00| 0.69|
|netral:|  0.00| 0.00| 0.00|
|positif:| 0.00|  0.00|  0.00|

Macro Avg F1-Score: 0.23

Weighted Avg F1-Score: 0.36

#### hasil 
- Overfitting ke Kelas Negatif:
   - Model hanya memprediksi "negatif" (recall 1.00) dan mengabaikan kelas lain (precision 0).
   - Akurasi 52.65% berasal dari dominasi kelas negatif dalam data (1053/2000 sampel).
- Gagal Total untuk Kelas Minoritas:
   - Precision dan recall 0% untuk "netral" dan "positif"
#### penyebab yang potensial
- Class Imbalance: Distribusi label tidak seimbang (negatif > positif > netral).
- Hyperparameter LSTM Kurang Optimal:
   - Dropout terlalu tinggi (0.5) atau unit LSTM terlalu sedikit (64).
- Data Tidak Terrepresentasi:
   - Kata kunci kelas minoritas tidak cukup dalam data training.



#### hasil dari cnn

Akurasi: 0.7990 (sekitar 80%)
Classification Report:

|kategori | precission | recall | f1-score |
|---------|---------|---------|----------|
|negatif: |0.87| 0.83| 0.85|
|netral: | 0.15 | 0.10 |0.12|
|positif: |0.79| 0.87| 0.83|

Macro Avg F1-Score: 0.60

Weighted Avg F1-Score: 0.79

#### hasil
- kinerja yang lebih baik:
   - Akurasi 79.9% dan F1-score tertimbang (weighted avg) 0.79.
   - Mampu mengenali pola "positif" (F1 0.83) dan "negatif" (F1 0.85).
- lebih robust:
   - Tidak overfit ke satu kelas (berbeda dengan LSTM).
- masih buruk untuk kelas netral
   - karena f1 score jumlah datanya hanya sedikit (130 sample)


 ## hasil keseluruhan 
 - CNN lebih unggul untuk kasus ini dengan akurasi 79.9% vs LSTM 52.65%.
 - LSTM gagal karena overfitting ke kelas mayoritas (negatif).
 - Netral adalah tantangan utama bagi kedua model (butuh lebih banyak data).
