# Laporan Proyek Machine Learning Recommendation System - Josua Sianturi

## Project Overview

Dengan banyaknya pilihan film yang tersedia di platform streaming, sistem rekomendasi menjadi sangat penting untuk membantu pengguna menemukan film yang sesuai dengan preferensi pribadi mereka. Banyak sistem saat ini mengandalkan **collaborative filtering**, namun pendekatan ini menghadapi masalah seperti **cold start** dan ketergantungan pada data histori interaksi pengguna.

Proyek ini menggunakan pendekatan **content-based filtering**, yang memanfaatkan fitur konten film, seperti **genre**, untuk merekomendasikan film yang mirip berdasarkan kesamaan kontennya (Fajriansyah, M.,dkk (2021)). Pendekatan ini memungkinkan sistem untuk merekomendasikan film baru atau kurang populer yang mungkin belum banyak mendapatkan rating dari pengguna lain.

Menurut penelitian oleh **Gomez-Uribe & Hunt (2016)**, metode berbasis konten efektif untuk meningkatkan relevansi rekomendasi dan memberikan pengalaman yang lebih personal bagi pengguna, terutama dalam mengatasi masalah **cold start** yang umum terjadi pada sistem berbasis rating.

Referensi :

- Fajriansyah, M., Adikara, P. P., & Widodo, A. W. (2021). Sistem Rekomendasi Film Menggunakan Content Based Filtering. Jurnal Pengembangan Teknologi Informasi Dan Ilmu Komputer, 5(6), 2188–2199. Diambil dari https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/9163

- Gomez-Uribe, C. A. & Hunt, N. 2016. *The Netflix Recommender System: Algorithms, Business Value, and Innovation*. ACM Transactions on Management Information Systems (TMIS), 6(4): Article 13. Tersedia secara daring di: [https://doi.org/10.1145/2843948](https://doi.org/10.1145/2843948)

---

## Business Understanding

### Problem Statements

1. **Pengguna kesulitan menemukan film baru yang sesuai dengan preferensi mereka.**  
   Dengan banyaknya pilihan film yang tersedia di platform streaming, pengguna sering merasa kewalahan dalam memilih tontonan yang relevan dengan minat pribadi mereka. Hal ini dapat mengurangi pengalaman menonton dan kepuasan pengguna.

2. **Masalah cold start pada sistem rekomendasi berbasis collaborative filtering.**  
   Pendekatan berbasis **collaborative filtering** bergantung pada histori interaksi pengguna dengan film, yang membuat sistem ini kurang efektif untuk pengguna baru atau film yang belum memiliki banyak rating. **Cold start** menghambat kemampuan sistem dalam memberikan rekomendasi yang relevan untuk pengguna baru atau film yang kurang populer.

3. **Kurangnya personalisasi berbasis konten film.**  
   Banyak sistem rekomendasi hanya menyarankan film berdasarkan kesamaan rating atau preferensi pengguna lain. Padahal, sistem rekomendasi berbasis **konten film**, seperti **genre**, dapat memperbaiki relevansi rekomendasi, bahkan untuk film yang kurang populer atau baru yang belum memiliki cukup banyak rating.


### Goals

1. **Membangun sistem rekomendasi berbasis konten untuk menyarankan film baru.**  
   Menggunakan pendekatan **content-based filtering**, sistem akan merekomendasikan film berdasarkan kesamaan **genre**. Pendekatan ini memungkinkan rekomendasi film baru atau film yang kurang populer tanpa memerlukan data rating dari pengguna lain.

2. **Memberikan rekomendasi yang lebih personal dan relevan.**  
   Rekomendasi akan disesuaikan dengan preferensi pengguna yang diidentifikasi melalui **genre** film yang telah mereka tonton sebelumnya. Sistem akan mencocokkan genre film yang disukai pengguna dengan film-film lain yang memiliki genre serupa.

3. **Meningkatkan interaksi pengguna dengan platform.**  
   Dengan meningkatkan relevansi rekomendasi film, diharapkan pengguna akan lebih sering berinteraksi dengan platform, meningkatkan frekuensi menonton, dan retensi pengguna.


### Solution Statements

**Pendekatan Content-Based Filtering**  
   Pendekatan ini menggunakan fitur **genre** film untuk menghitung kemiripan antar film. Teknik **cosine similarity** atau algoritma serupa akan digunakan untuk mengukur kesamaan antara film yang sudah diberi rating tinggi oleh pengguna dan film lainnya berdasarkan genre yang dimiliki. Sistem ini memungkinkan rekomendasi film yang relevan meskipun film tersebut belum banyak mendapatkan rating.

---

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **MovieLens (Small) Dataset**, yang berisi data mengenai film, rating yang diberikan oleh pengguna, serta informasi terkait lainnya. Dataset ini dapat diunduh dari situs resmi GroupLens.

### Sumber Data

Dataset **MovieLens (Small)** dapat diunduh dari situs resmi GroupLens di [tautan berikut](https://grouplens.org/datasets/movielens/) (pilih versi `ml-latest-small.zip`).

### Uraian Seluruh Variabel atau Fitur pada Data

Dataset **MovieLens (Small)** yang digunakan terdiri dari dua file utama, yaitu **ratings.csv** dan **movies.csv**. Berikut adalah penjelasan tentang fitur yang ada dalam masing-masing file:

#### **ratings.csv**

- **userId**: Identifikasi unik untuk setiap pengguna. Angka ini digunakan untuk membedakan satu pengguna dengan yang lainnya.
- **movieId**: Identifikasi unik untuk setiap film. Kolom ini digunakan untuk mengidentifikasi film yang diberikan rating oleh pengguna.
- **rating**: Nilai numerik rating yang diberikan oleh pengguna untuk sebuah film, dalam skala 0.5 hingga 5.0. Rating ini menggambarkan seberapa besar minat pengguna terhadap film tersebut.
- **timestamp**: Waktu ketika rating diberikan, dalam format Unix timestamp. Kolom ini menunjukkan kapan pengguna memberikan rating terhadap film tersebut.

#### **movies.csv**

- **movieId**: Identifikasi unik untuk setiap film. Kolom ini digunakan untuk menghubungkan data film dengan data rating melalui `movieId`.
- **title**: Judul film (misalnya, "Toy Story (1995)"). Kolom ini mencatat nama lengkap film yang terkait dengan `movieId`.
- **genres**: Kategori genre film yang terkait, dipisahkan oleh karakter "|" (misalnya, "Adventure|Animation|Children|Comedy|Fantasy"). Kolom ini mencakup satu atau lebih genre yang relevan dengan film.

### Data Condition

- Dataset **ratings.csv** terdiri dari **100,836 baris** data, dan **movies.csv** terdiri dari **9,742 baris** data.
- Tidak terdapat nilai kosong dalam kolom `userId`, `movieId`, `rating`, dan `timestamp` pada **ratings.csv**. Demikian juga Kolom `movieId`, `title`, dan `genres` pada **movies.csv**.

---

## Data Preparation

### Melihat Missing Value

Dilakukan pengecekan terhadap keberadaan nilai yang hilang (missing value) pada dataset `movies_df`.  
Hasilnya menunjukkan tidak ada missing value, sehingga seluruh kolom data lengkap dan siap diproses.

```python
movies_df.isnull().sum()

```
### Melihat Data Duplikat

Untuk memastikan bahwa setiap entri film dalam dataset `movies_df` bersifat unik, dilakukan pengecekan terhadap data duplikat.  
Hasil pemeriksaan menunjukkan tidak ada baris data yang duplikat, sehingga semua data film dapat digunakan tanpa risiko pengulangan yang tidak diinginkan.

```python
movies_df.duplicated().sum()
```

### Menghapus Film Tanpa Informasi Genre

Film yang tidak memiliki informasi genre (`'(no genres listed)'`) dihapus dari dataset karena tidak memberikan fitur yang berguna dalam sistem rekomendasi berbasis konten yang mengandalkan genre sebagai dasar kemiripan antar film.  
Dengan menghilangkan film-film tersebut, data yang digunakan menjadi lebih bersih dan relevan untuk pemodelan.

```python
content_data = movies_df[movies_df['genres'] != '(no genres listed)'].copy()
content_data.head()
```

### Menangani Genre Bertanda Hubung agar Tidak Terpecah oleh TF-IDF

Secara default, `TfidfVectorizer` memisahkan kata berdasarkan spasi dan tanda baca, termasuk tanda hubung (`-`).  
Hal ini menyebabkan genre seperti `"sci-fi"` atau `"film-noir"` terpecah menjadi token yang tidak representatif seperti `"sci"`, `"fi"`, `"film"`, dan `"noir"`.

Untuk mengatasi masalah ini, kita mengganti tanda hubung (`-`) menjadi garis bawah (`_`) pada kolom genre.  
Dengan demikian, genre yang terdiri dari dua kata tetap dianggap sebagai satu token, sehingga representasi fitur menjadi lebih akurat.

```python
content_data['genres'] = content_data['genres'].str.replace('-', '_')
```

### Menghitung Jumlah Film Setelah Filtering

Kode ini menghitung jumlah film yang tersisa setelah menghapus film tanpa genre `(no genres listed)`.  
Angka ini menunjukkan berapa banyak data film valid yang siap digunakan untuk proses pembuatan sistem rekomendasi.

```python
len(content_data)
```

### TF-IDF Vectorization

Pada tahap ini, kita mengubah informasi genre dari dataset MovieLens yang bersifat teks menjadi representasi numerik menggunakan teknik **TF-IDF (Term Frequency-Inverse Document Frequency)**. 

**TF-IDF** adalah metode yang mengukur seberapa penting suatu kata dalam sebuah dokumen relatif terhadap keseluruhan kumpulan dokumen. **Term Frequency (TF)** mengukur frekuensi kemunculan kata dalam dokumen, sementara **Inverse Document Frequency (IDF)** mengukur seberapa jarang kata tersebut muncul di seluruh dokumen. Proses ini memberikan bobot lebih pada kata-kata yang sering muncul dalam dokumen tertentu namun jarang di seluruh dataset, sehingga menghasilkan representasi yang lebih relevan untuk sistem rekomendasi.

Proses ini dilakukan melalui langkah-langkah berikut:
1. Inisialisasi TfidfVectorizer  
   Menggunakan `TfidfVectorizer` dari pustaka `sklearn` untuk mengonversi data teks genre film menjadi format numerik.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Inisialisasi TfidfVectorizer
tf = TfidfVectorizer()
```

2. Fitting dan Transformasi Data  
   Fungsi `fit()` digunakan untuk mempelajari genre-genre unik yang ada pada dataset, sedangkan `transform()` menghasilkan matriks TF-IDF yang merepresentasikan film berdasarkan genre mereka.  

   Langkah pertama, `fit()` mempelajari distribusi genre di seluruh dataset, mengenali kata kunci genre yang unik. Setelah itu, `transform()` diterapkan pada data untuk menghasilkan matriks yang menunjukkan seberapa sering dan seberapa penting suatu genre muncul pada setiap film dalam dataset.

```python
# Melakukan fit pada kolom 'genres'
tf.fit(content_data['genres'])
# Melakukan transformasi ke matrix TF-IDF
tfidf_matrix = tf.transform(content_data['genres'])

```

3. Sparse to Dense Matrix  
   Matriks TF-IDF yang awalnya sparse diubah menjadi matriks penuh (dense) untuk mempermudah analisis lebih lanjut.  

   Matriks sparse hanya menyimpan nilai-nilai yang tidak nol, sehingga menghemat ruang memori. Namun, untuk tujuan inspeksi dan analisis manual, matriks dense (yang menyimpan semua nilai, termasuk nol) lebih mudah dibaca dan dipahami.

   Pada langkah ini, kita mengubah matriks TF-IDF menjadi matriks penuh menggunakan metode `.todense()`.

```python
tfidf_matrix.todense()
```
TF-IDF Vectorization ini penting karena mengubah data teks menjadi representasi numerik menggunakan TF-IDF, yang memungkinkan sistem rekomendasi menghitung kemiripan antar film dengan memberikan bobot lebih pada genre relevan, serta mempersiapkan data untuk perhitungan kemiripan berbasis konten.

---

## Modeling and Result

Sistem rekomendasi film ini dibangun menggunakan pendekatan **Content-Based Filtering**, dengan memanfaatkan informasi **genre** sebagai fitur utama dalam menentukan kemiripan antar film.

#### Apa itu Content-Based Filtering?
**Content-Based Filtering** adalah pendekatan sistem rekomendasi yang menyarankan item (film, lagu, buku, dll.) kepada pengguna berdasarkan kesamaan karakteristik konten item tersebut. Dalam kasus ini, sistem menganalisis fitur-fitur yang melekat pada item — seperti *genre* film — untuk menemukan item lain yang serupa.

Pendekatan ini tidak memerlukan data perilaku pengguna lain, melainkan hanya bergantung pada fitur deskriptif dari item yang bersangkutan.

### Algoritma yang Digunakan

Model ini menggunakan kombinasi dua metode utama:

1. **Cosine Similarity**  
   Setelah genre diubah menjadi representasi numerik melalui **TF-IDF Vectorization**(yang telah dilakukan di Tahap Data Preparation), kemiripan antar film dihitung menggunakan **Cosine Similarity**. Nilai similarity ini menunjukkan seberapa mirip dua film berdasarkan genre-nya. Nilai cosine similarity berkisar antara 0 (tidak mirip) hingga 1 (sangat mirip).

2. **Fungsi Rekomendasi**  
   Fungsi `recommend_similar_movies()` dibuat untuk memberikan daftar film yang paling mirip dengan film input berdasarkan genre. Fungsi ini memanfaatkan matriks cosine similarity untuk menentukan film yang paling relevan.

### Langkah-Langkah Modeling

- **Perhitungan Cosine Similarity**:  
  Matriks cosine similarity dihitung untuk semua pasangan film, menghasilkan tabel kemiripan yang digunakan dalam sistem rekomendasi.

- **Fungsi Rekomendasi**:  
  Fungsi `recommend_similar_movies()` digunakan untuk memberikan daftar film yang paling mirip dengan film input berdasarkan genre.


### Result
Berikut adalah daftar 10 film teratas yang direkomendasikan karena memiliki genre yang sama atau sangat mirip dengan film referensi **"Me Myself I (2000)"**, yang bergenre **Comedy** dan **Romance**:

#### Top 10 Rekomendasi Film Mirip *Me Myself I (2000)* (Content-Based)

| No | Judul Film                                       | Genre            | Similarity Score |
|----|--------------------------------------------------|------------------|------------------|
| 0  | Sabrina (1995)                                   | Comedy\|Romance  | 1.0              |
| 1  | Two if by Sea (1996)                             | Comedy\|Romance  | 1.0              |
| 2  | French Twist (Gazon maudit) (1995)               | Comedy\|Romance  | 1.0              |
| 3  | Bedrooms & Hallways (1998)                       | Comedy\|Romance  | 1.0              |
| 4  | Black Cat, White Cat (Crna macka, beli macor)    | Comedy\|Romance  | 1.0              |
| 5  | Lady Eve, The (1941)                             | Comedy\|Romance  | 1.0              |
| 6  | Fever Pitch (1997)                               | Comedy\|Romance  | 1.0              |
| 7  | Three to Tango (1999)                            | Comedy\|Romance  | 1.0              |
| 8  | Bachelor, The (1999)                             | Comedy\|Romance  | 1.0              |
| 9  | Moonstruck (1987)                                | Comedy\|Romance  | 1.0              |

> Semua film di atas memiliki kemiripan maksimal (1.0) terhadap film referensi karena genre-nya identik, sehingga dianggap sangat relevan dalam konteks content-based filtering berbasis genre.

### Kelebihan Pendekatan Ini

- Tidak memerlukan data pengguna, sistem dapat memberikan rekomendasi meskipun tanpa informasi riwayat interaksi pengguna (seperti rating atau riwayat tontonan).
- Cocok untuk cold-start item, dapat merekomendasikan film yang baru ditambahkan ke sistem asalkan memiliki informasi genre.

### Kelemahan

- Kurang personalisasi, Sistem tidak mempertimbangkan preferensi pengguna tertentu.
- Tidak mempertimbangkan popularitas atau feedback pengguna.

---

##  Evaluation

**Precision@K** adalah metrik evaluasi yang digunakan untuk mengukur akurasi rekomendasi dalam sistem rekomendasi. Metrik ini mengukur proporsi film yang relevan dari seluruh rekomendasi yang diberikan dalam **top K** rekomendasi.

**Relevansi** dalam konteks ini didefinisikan sebagai kemiripan genre antara film yang direkomendasikan dan film yang diinginkan oleh pengguna (input).

Rumus untuk menghitung **Precision@K** adalah sebagai berikut:

$$
\text{Precision@K} = \frac{\text{Jumlah Film Relevan dalam Top K}}{K}
$$

Dimana:
- **Jumlah Film Relevan dalam Top K** adalah jumlah film yang memiliki kesamaan genre dengan film yang diinginkan dalam daftar rekomendasi.
- **K** adalah jumlah rekomendasi yang diberikan.

Untuk Pendekatan Content-Based Filtering pada kasus ini kita bisa menghitung Precision@K secara manual dengan melihat daftar 10 film yang direkomendasikan oleh sistem.

Hasil Evaluasi:
- Jumlah Film Relevan (R): 10 (semua film memiliki genre yang relevan),
- Precision@10 = 10/10 = 1.0

Karena semua film yang direkomendasikan memiliki genre yang relevan (Comedy|Romance), sistem mencapai **Precision@10 = 1.0**, yang menunjukkan performa sistem yang sangat baik dalam memberikan rekomendasi yang relevan, dengan demikian sistem ini cocok untuk pengguna yang menginginkan rekomendasi berdasarkan **kesamaan genre** dan dapat memberikan hasil yang sangat relevan.

---

##  Keterkaitan dengan Business Understanding

Evaluasi model ini menunjukkan bahwa sistem rekomendasi yang dikembangkan telah **berdampak langsung terhadap pemenuhan Business Understanding**, dengan penjelasan sebagai berikut:

### Apakah model menjawab setiap *Problem Statement*?

1. **Pengguna kesulitan menemukan film yang sesuai preferensi**  
   *Terjawab*: Sistem mampu merekomendasikan film dengan genre yang serupa dengan film yang disukai pengguna, sehingga membantu mereka menemukan tontonan baru yang relevan.

2. **Masalah cold start pada collaborative filtering**  
   *Teratasi*: Karena menggunakan pendekatan content-based filtering berbasis genre, sistem tetap dapat merekomendasikan film meskipun tidak ada histori rating pengguna atau film tersebut belum populer.

3. **Kurangnya personalisasi berbasis konten film**  
   *Terjawab*: Rekomendasi dihasilkan dari kesamaan konten (genre), bukan hanya kesamaan perilaku pengguna lain, sehingga lebih personal.

### Apakah model berhasil mencapai setiap *Goals* yang diharapkan?

1. **Membangun sistem content-based yang menyarankan film baru**  
   *Tercapai*: Sistem mampu merekomendasikan film yang tidak harus populer atau memiliki banyak rating, selama memiliki kesamaan konten.

2. **Memberikan rekomendasi yang lebih personal dan relevan**  
   *Tercapai*: Rekomendasi disesuaikan dengan film yang pernah ditonton pengguna, berdasarkan genre-nya.

3. **Meningkatkan interaksi pengguna dengan platform**  
   *Berpotensi tercapai*: Dengan meningkatkan relevansi rekomendasi, pengguna lebih mungkin untuk menemukan film menarik, yang mendorong interaksi lebih lanjut.

### Apakah *Solution Statement* berdampak?

Pendekatan **content-based filtering dengan cosine similarity terhadap genre film** terbukti efektif:
- Memberikan hasil yang konsisten dan relevan.
- Mengatasi keterbatasan collaborative filtering pada pengguna/film baru.
- Mudah diimplementasikan dan skalabel.

Namun, hasil similarity yang terlalu tinggi dan seragam (skor 1.0 untuk semua rekomendasi) juga menunjukkan keterbatasan jika hanya mengandalkan genre. Oleh karena itu, peningkatan kualitas dapat dicapai dengan menggabungkan fitur konten lain (sinopsis, aktor, sutradara) dan mengadopsi pendekatan **hybrid filtering**.

---

## Kesimpulan

Model yang dikembangkan telah secara efektif menjawab tantangan bisnis yang diidentifikasi, dan mampu memenuhi tujuan proyek dengan pendekatan yang sesuai. Hasil evaluasi menggunakan metrik Precision@10 menunjukkan performa yang sangat baik, dengan seluruh film dalam daftar rekomendasi memiliki genre yang relevan dengan film target. Hal ini menunjukkan bahwa sistem mampu memahami kesamaan konten secara efektif,dan sistem memberikan rekomendasi yang relevan secara konten dan personal. Untuk pengembangan lanjutan, sistem dapat ditingkatkan dengan memperluas fitur konten dan mengadopsi pendekatan hybrid untuk hasil yang lebih bervariasi dan kontekstual.


