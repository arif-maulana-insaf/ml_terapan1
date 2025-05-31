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
#### Solusi 1: Model LSTM dengan Hyperparameter Tuning
- Alasan: LSTM terbukti efektif untuk teks bahasa Indonesia (F1-score 0.82 dalam penelitian Suryanto et al., 2022).
- Improvement:
  - Optimasi hyperparameter:
```python
model = Sequential([
    Embedding(vocab_size, 256),  
    Bidirectional(LSTM(128, return_sequences=True)),  
    Dropout(0.5),  
    GlobalMaxPooling1D(),  
    Dense(64, activation='relu'),  
    Dense(3, activation='softmax')  
])
```

Optimasi hyperparameter:
Metrik Evaluasi:
Akurasi
F1-score (untuk handle class imbalance)
AUC-ROC (jika threshold adjustment diperlukan)

#### Solusi 2: Fine-Tuning Model Pre-trained Bahasa Indonesia (IndoBERT)
- Alasan: IndoBERT telah dilatih pada korpus bahasa Indonesia yang luas (Wilie et al., 2020).
- Implementasi:
```python
from transformers import BertTokenizer, TFBertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = TFBertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p1', num_labels=3)
```
- Metrik Evaluasi:
- Precision per kelas (Positif/Negatif/Netral)
- Recall untuk kelas minoritas (biasanya negatif)

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
| # | co;umn | non-null count | Dtype |
| 1 | userName | 10000 non-null | object |
| 2 | cotent | 10000 non-null | object |
| 3 | score | 10000 non-null | int | 
| 4 | at | 10000 non-null | object | 
| 4 | appVersion | 8026 non-null | object |
