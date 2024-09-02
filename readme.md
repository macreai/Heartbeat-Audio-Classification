
## Domain Proyek

Menurut Organisasi Kesehatan Dunia (WHO), Penyakit kardiovaskular (CVD) adalah penyebab kematian utama di seluruh dunia, menyebabkan sekitar 17,9 juta kematian setiap tahunnya. CVD mencakup berbagai gangguan pada jantung dan pembuluh darah, termasuk penyakit jantung koroner, penyakit serebrovaskular, penyakit jantung rematik, dan kondisi lainnya. Lebih dari empat dari lima kematian akibat CVD disebabkan oleh serangan jantung dan stroke, dan sepertiga dari kematian tersebut terjadi pada orang yang belum mencapai usia 70 tahun. [1]

Dalam beberapa tahun terakhir, kemajuan dalam penelitian machine learning telah memberikan harapan baru untuk deteksi dini penyakit jantung. Teknologi ini berpotensi meningkatkan identifikasi penyakit kardiovaskular secara lebih awal, yang dapat mengurangi angka kematian dan meningkatkan hasil kesehatan. Namun, masih ada tantangan signifikan yang perlu diatasi. Salah satunya adalah akurasi identifikasi yang sering kali kurang optimal, dengan banyak sistem yang menghasilkan akurasi di bawah 90%. Hal ini mengarah pada kemungkinan mengurangi keandalan sistem dalam praktek klinis.

Tantangan lainnya adalah masalah denoising, atau pengurangan gangguan dalam data yang digunakan oleh model machine learning. Data medis sering kali terpengaruh oleh noise atau gangguan, seperti kesalahan pengukuran atau variabilitas dalam sinyal medis. Ketika proses denoising kurang efektif, informasi yang relevan bisa hilang atau menjadi tidak jelas, sehingga mengurangi kemampuan model untuk melakukan prediksi yang akurat. Proses pelatihan model yang kompleks dan pengolahan data intensif memerlukan perangkat keras canggih serta biaya yang tinggi, yang sering kali membatasi penerapannya di fasilitas kesehatan dengan anggaran terbatas. Untuk memajukan teknologi ini, perlu ada upaya untuk meningkatkan akurasi algoritma, mengatasi masalah denoising, dan mengurangi biaya komputasi, misalnya melalui pengembangan model yang lebih efisien. Dengan mengatasi tantangan ini, pembelajaran mesin dapat lebih efektif dalam mendeteksi dan mengelola penyakit jantung, memberikan manfaat yang lebih besar dalam meningkatkan kesehatan global.

## Business Understanding

### Problem Statements

- Performa Pengidentifikasian Model Tidak Optimal
  - Model yang digunakan untuk pengidentifikasian memiliki performa yang kurang optimal, yang dapat mengakibatkan hasil yang tidak akurat dan tidak efisien.

- Noise yang Terdapat pada Data Mengganggu Performa Model
  - Adanya noise atau gangguan dalam data yang digunakan mengakibatkan penurunan akurasi dan keandalan model, sehingga mengganggu hasil analisis dan prediksi.

### Goals

- Merancang model klasifikasi dengan batas minimum sensitivitas 90%
- Mengurangi atau menghilangkan noise dari data melalui teknik audio denoising, sehingga meningkatkan kualitas data dan performa model.

### Solution Statement

- Menerapkan Audio Denoising dengan Alat Ukur STOI
  - Tujuan: Mengurangi noise pada data audio menggunakan teknik denoising, diukur dengan alat ukur STOI (Short-Time Objective Intelligibility). STOI mengukur intelligibility (keterbacaan) audio dengan membandingkan kualitas audio bersih dengan audio yang telah terdistorsi, memastikan bahwa data yang digunakan dalam model lebih bersih dan relevan.[2]

- Menerapkan Feature Engineering pada Data Audio
  - Tujuan: Meningkatkan kualitas data dan performa model dengan menciptakan fitur baru yaitu dengan menggunakan statistical fitur yang relevan dari data audio. Ini akan membantu model dalam menemukan pola yang lebih kompleks dan mengatasi masalah data yang tidak seimbang.[3]

- Menerapkan PCA
  - Tujuan: Menggunakan PCA untuk mengurangi jumlah fitur dalam dataset dengan mempertahankan sebanyak mungkin variansi informasi asli. Ini membantu dalam mempermudah model, mempercepat waktu komputasi, dan mengurangi risiko overfitting dengan menyederhanakan data tanpa kehilangan informasi penting.

- Menerapkan Hyperparameter tuning dengan GridSearch
  - Tujuan: Menggunakan GridSearch untuk mengeksplorasi berbagai kombinasi hyperparameter model secara sistematis, sehingga dapat menemukan konfigurasi yang memberikan performa terbaik berdasarkan metrik evaluasi yang ditentukan yaitu akurasi.

- Menerapakan K-Fold Cross Validation Sebanyak 7
  - Tujuan: Dengan menerapkan 7-Fold Cross Validation, model dievaluasi pada 7 subset berbeda dari data, yang membantu mengurangi bias dan variansi hasil evaluasi, sehingga memberikan estimasi yang lebih akurat tentang performa model pada data yang tidak terlihat. Pengguaan sebanyak 7 Fold juga terbukti lebih baik dibandingkan jumlah fold lainnya.[4]

- Membandingkan Algoritma Klasifikasi dengan Sensitivitas
  - Tujuan: Mengukur dan membandingkan sensitivitas berbagai tes atau model untuk menentukan seberapa efektif mereka dalam mendeteksi penyakit atau kondisi tertentu. Sensitivitas mengukur proporsi kasus positif yang benar.[5]

## Data Understanding #1
Data telah dikumpulkan dari masyarakat umum melalui aplikasi iStethoscope Pro di iPhone dan dari uji klinis di rumah sakit menggunakan stetoskop digital DigiScope. Versi asli dari dataset hanya memiliki 585 sampel audio, tetapi data ini berhasil mengumpulkan dan memberi label 76 sampel audio dari berbagai sumber untuk meningkatkan representasi dan mengurangi dampak ketidakseimbangan kelas. Totalnya ada 661 file audio yang diberi label kelas di folder audio. Folder yang tidak diberi label berisi 247 file audio yang belum diberi tanda. Mereka dikecualikan dari file markup untuk menghindari kebingungan karena mereka digunakan untuk menguji model dan mengirimkan hasil. Folder info hanya berisi informasi tentang dataset termasuk deskripsi saat ini, lisensi, dan gambar dengan grafik kelas serta informasi lainnya.

Terdapat 5 Label kelas pada dataset yang digunakan ini, berikut adalah penjelasan masing-masing Label kelas:

#### Kategori Normal
- **Suara jantung normal:** Suara jantung sehat yang memiliki pola "lub dub, lub dub" yang jelas.
- **Kebisingan:** Dapat berisi suara latar belakang seperti lalu lintas atau radio, dan kebisingan acak seperti napas atau gesekan mikrofon.
- **Keterangan temporal:** Waktu antara "lub" dan "dub" lebih pendek daripada waktu antara "dub" dan "lub" berikutnya.
- **Denyut jantung:** Biasanya antara 60 hingga 100 kali per menit saat istirahat, namun bisa bervariasi antara 40 hingga 140 atau lebih per menit.
- **Noisy Normal:** Data normal dengan gangguan kebisingan yang substansial.

#### Kategori Murmur
- **Suara Murmur Jantung:** Suara seperti “whooshing, roaring, rumbling, or turbulent fluid” yang muncul di antara "lub" dan "dub" atau antara "dub" dan "lub".
- **Gejala:** Bisa menjadi indikasi berbagai kelainan jantung, beberapa di antaranya serius.
- **Keterangan temporal:** Murmur terjadi antara "lub" dan "dub" atau antara "dub" dan "lub", bukan pada "lub" atau "dub".

#### Kategori Suara Jantung Tambahan (Extra Heart Sound)
- **Suara tambahan:** Suara tambahan seperti “lub-lub dub” atau “lub dub-dub”.
- **Keterangan:** Suara tambahan bisa menjadi tanda penyakit, namun juga bisa normal pada beberapa situasi.
- **Keterangan temporal:** Suara tambahan terjadi setelah "lub" atau "dub", menciptakan pola "lub-lub dub" atau "lub dub-dub".

#### Kategori Artefak (Artifact)
- **Suara Artefak:** Suara seperti umpan balik, gema, suara manusia, musik, atau kebisingan lainnya yang tidak memiliki pola jantung yang teratur.
- **Frekuensi:** Tidak memiliki periodisitas temporal pada frekuensi di bawah 195 Hz.
- **Penting:** Memisahkan kategori ini dari yang lain untuk memastikan pengambilan data yang lebih baik.

#### Kategori Extrasystole
- **Suara Extrasystole:** Suara jantung yang tidak berirama dengan tambahan atau detak yang terlewat, seperti “lub-lub dub” atau “lub dub-dub”.
- **Keterangan:** Bisa normal, terutama pada anak-anak, namun juga bisa menjadi tanda penyakit jantung dalam beberapa situasi.
- **Keterangan temporal:** Peristiwa ini tidak terjadi secara teratur dan mungkin menunjukkan detak jantung yang tidak normal.

Dataset: [6] & [7]

Download dataset: [Dangerous Heartbeat Dataset](https://www.kaggle.com/datasets/mersico/dangerous-heartbeat-dataset-dhd)

Terdapat metadata sebagai informasi dari Dangerous Heartbeat Dataset, sebagai berikut:   

### Variabel-variabel pada metadata DHD dataset adalah sebagai berikut:
- sample_rate : Frekuensi sampling audio dalam Hertz (Hz)
- num_frames : Jumlah total frame atau sampel dalam file audio.
- num_channels : Jumlah saluran audio dalam file. Nilai 1 menunjukkan audio mono (satu saluran), sedangkan nilai 2 menunjukkan audio stereo (dua saluran).
- bits_per_sample : Jumlah bit yang digunakan untuk mewakili setiap sampel audio.
- encoding : Format encoding atau kompresi dari data audio.
- duration : Durasi file audio dalam detik.
- filename : Nama file dari sampel audio.

Juga terdapat labels sebagai informasi label dari Dangerous Heartbeat Dataset, sebagai berikut:

### Variabel-variable pada labels DHD dataset adalah sebagai berikut:
- set : Metode yang digunakan dalam mengambil audio.
- filename : Nama file dari sampel audio.
- label : Label kelas

**Dataset yang digunakan dalam penelitian ini terdiri dari 661 entri, di mana setiap entri mewakili satu file audio. Dataset ini memiliki tujuh kolom utama:**

- **sample_rate**: Kolom ini mencatat laju pengambilan sampel untuk setiap file audio dalam dataset. Semua 661 entri memiliki nilai sample_rate, dan data tipe yang digunakan adalah int64.

- **num_frames**: Kolom ini mencatat jumlah frame yang ada dalam setiap file audio. Sama seperti kolom sebelumnya, setiap entri dalam dataset memiliki nilai pada kolom ini, dengan tipe data int64.

- **num_channels**: Kolom ini menunjukkan jumlah saluran (channel) audio yang terdapat dalam file, misalnya, apakah file audio tersebut stereo atau mono. Data ini juga tercatat lengkap untuk semua entri dengan tipe data int64.

- **bits_per_sample**: Kolom ini mencatat jumlah bit yang digunakan untuk setiap sampel dalam file audio. Sama dengan kolom lainnya, setiap entri memiliki nilai bits_per_sample dengan tipe data int64.

- **encoding**: Kolom ini berisi informasi mengenai jenis pengkodean yang digunakan dalam file audio, seperti PCM (Pulse Code Modulation) atau lainnya. Data pada kolom ini bertipe object, dan tidak ada nilai yang hilang.

- **duration**: Kolom ini mengukur durasi dari setiap file audio dalam satuan detik. Kolom ini menggunakan tipe data float64 dan tidak ada nilai yang hilang di seluruh dataset.

- **filename**: Kolom terakhir mencatat nama file dari setiap entri audio. Data ini menggunakan tipe object dan memiliki nilai untuk setiap entri.

**Secara keseluruhan, dataset ini memiliki total 661 entri dengan berbagai karakteristik teknis dari file audio yang tercatat lengkap tanpa ada data yang hilang. Data ini akan digunakan untuk analisis lebih lanjut dan feature engineering dalam rangka mengembangkan model klasifikasi audio.**


### Exploratory Data Analysis

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sample_rate</th>
      <th>num_frames</th>
      <th>num_channels</th>
      <th>bits_per_sample</th>
      <th>duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>661.000000</td>
      <td>6.610000e+02</td>
      <td>661.000000</td>
      <td>661.0</td>
      <td>661.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>15757.791225</td>
      <td>2.196370e+05</td>
      <td>1.083207</td>
      <td>16.0</td>
      <td>9.066551</td>
    </tr>
    <tr>
      <th>std</th>
      <td>18400.922135</td>
      <td>6.313720e+05</td>
      <td>0.276404</td>
      <td>0.0</td>
      <td>13.263770</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4000.000000</td>
      <td>3.053000e+03</td>
      <td>1.000000</td>
      <td>16.0</td>
      <td>0.760000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4000.000000</td>
      <td>1.504600e+04</td>
      <td>1.000000</td>
      <td>16.0</td>
      <td>3.620000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4000.000000</td>
      <td>3.522900e+04</td>
      <td>1.000000</td>
      <td>16.0</td>
      <td>7.390000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>44100.000000</td>
      <td>3.309340e+05</td>
      <td>1.000000</td>
      <td>16.0</td>
      <td>9.770000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>48000.000000</td>
      <td>1.008000e+07</td>
      <td>2.000000</td>
      <td>16.0</td>
      <td>210.000000</td>
    </tr>
  </tbody>
</table>
</div>

**Terlihat adanya ketidakseragaman pada fitur sample_rate dan num_channels**

### Visualisasi Data

![label_raw_bar_plot](https://github.com/user-attachments/assets/930902d8-2e01-46f9-9d61-1354e11b0b1a)<br>
![label_raw_pie_chart](https://github.com/user-attachments/assets/f8e77d88-0bff-4b25-80a9-cea9ddcedb06)

**Label yang ada pada data penyakit jantung ini terdiri dari Normal, murmur, artifact, extrastole, dan extrahls. Namun, data yang diterima tidak seimbang terlihat Normal lebih dominan.**

![samplerate_raw_bar_plot](https://github.com/user-attachments/assets/89fae8b0-cc7c-4d3f-8231-775acfc8fe15)<br>
![samplerate_raw_pie_chart](https://github.com/user-attachments/assets/1403f3d1-e870-4a55-98b1-8f2f978bc46c)

**Terdapat beberapa jenis sample rate pada data audio, kita akan mengonversi semuanya ke 44100 agar ekstraksi fitur yang kita lakukan memiliki informasi fitur sample rate yang sama**

![channel_raw_bar_plot](https://github.com/user-attachments/assets/5aeb23d3-7794-47ed-8118-9b196a30ce12)<br>
![channel_raw_pie_chart](https://github.com/user-attachments/assets/ad7c9462-7658-457e-ab28-0d6dc27a2a89)

**Ditemukan data Wav dengan channel stereo (2) sementara kita hanya memerlukan data mono saja (1) kita akan mengonversi data wav tersebut ke mono agar ekstraksi fitur yang kita lakukan memiliki informasi fitur yang sama yaitu mono**

## Data Preprocessing #1
**Melakukan konversi atau resampling rate audio ke 44100 Hz untuk semua data audio agar ekstraksi fitur memiliki sample rate yang konsisten. Proses ini dilakukan dengan bantuan library librosa, dan hasilnya disimpan dalam direktori terpisah**

**Berikut adalah hasil konversi audio:**

![samplerate_converted](https://github.com/user-attachments/assets/c6264cc4-5d82-49d5-b634-d372e3359ca7)

**Visualisasi di atas menunjukkan bahwa semua audio telah berhasil dikonversi ke 44100 Hz.**

**Selanjutnya, kita akan melakukan konversi channel audio dari stereo ke mono untuk memastikan bahwa ekstraksi fitur yang dilakukan memiliki format yang seragam (mono). Hasil konversi ini juga disimpan dalam direktori terpisah**

**Berikut adalah hasil konversi audio dari stereo ke mono:**

![channel_converted](https://github.com/user-attachments/assets/67daa749-20fc-468a-9d71-8d0b555a7340)

**Dari visualisasi di atas, dapat dilihat bahwa semua channel audio telah berhasil dikonversi ke format Mono.**

### Denoising
**Pada tahap denoising audio ini, saya menggunakan algoritma FIR filter. Terdapat studi yang membandingkan performa FIR dan IIR filter, dan hasil studi tersebut menunjukkan bahwa FIR filter dengan Hamming window memberikan performa yang lebih baik. Filter ini menggunakan frekuensi cut-off sebesar 4000 Hz dan order 100.[9]** 

**Berikut adalah cuplikan kode FIR Filter dengan frekuensi cut-off sebesar 4000 Hz dan order 100 dengan bantuan library scipy:**

```
def audio_filter_FIR(audio_file, output_file):
    data, sample_rate = sf.read(audio_file)

    cutoff_freq = 4000.0
    filter_order = 100

    nyquist_rate = sample_rate / 2.0
    normalized_cutoff = cutoff_freq / nyquist_rate

    fir_coefficients = signal.firwin(filter_order, normalized_cutoff, window='hamming')

    filtered_data = signal.lfilter(fir_coefficients, 1.0, data)
    sf.write(output_file, filtered_data, sample_rate)
```

**Dengan memanfaatkan pengukuran STOI, saya mengukur kualitas audio yang telah dibersihkan. Caranya adalah dengan mengomputasikan nilai STOI untuk semua file audio, kemudian merata-ratakannya dengan bantuan library pystoi.**

**Nilai STOI berkisar antara 0 hingga 1, di mana nilai yang lebih tinggi menunjukkan performa yang lebih baik. Hasil pengukuran STOI yang diperoleh adalah 0.9202424067554108.**

## Perbandingan antara data audio raw dan data audio denoised

**Pada tahap ini kita membandingkan sebelum dan sesudah audio dibersihkan**

![comparison_1](https://github.com/user-attachments/assets/c570b60a-d659-49dc-9330-de81219c3aa2)

**Mungkin terlihat bahwa raw audio dan denoised audio tidak berbeda, namun perhatikan pada detik ke-10 perbedaan begitu terlihat. Data yang digunakan merupakan data akademis yang sudah bersih, namun tetap dilakukan denoising untuk meningkatkan performa model yang akan dibuat dan simulasi pada data raw yang akan memiliki noise saat deployment. Mari kita lanjutkan pada data audio berikutnya**

![comparison_2](https://github.com/user-attachments/assets/569331b6-3279-4f55-9766-2041f4ce8bdd)

**Pada data sampel kali ini pun data denoised dan data raw terlihat tidak jauh berbeda, hal ini dibuktikan dari data raw yang memang sudah bersih**

### Feature Engineering
**Pada tahap ini, saya melakukan feature engineering untuk meningkatkan kualitas data dan performa model dengan menciptakan fitur-fitur baru yang dapat menangkap informasi penting dari data audio. Feature engineering ini bertujuan untuk mengubah sinyal audio menjadi representasi numerik yang lebih komprehensif, yang dapat digunakan oleh model untuk mendeteksi pola-pola spesifik.[3]** <br>

**Fitur yang diambil meliputi:**

- **Mel-frequency cepstral coefficients (MFCCs)**
  <br>
  Fitur ini dipilih karena MFCCs secara efektif merepresentasikan karakteristik frekuensi dari sinyal audio, yang penting dalam pengenalan pola suara.
- **Chromagram from a waveform or power spectrogram (Chroma STFTs)**
  <br>
  Fitur ini membantu dalam menangkap informasi terkait nada dan tonalitas dalam sinyal audio.
- **Constant-Q chromagram (Chroma CQTs)**
  <br>
  Fitur ini digunakan untuk analisis tonal dalam domain frekuensi, yang memberikan stabilitas lebih baik pada perubahan skala frekuensi.
- **Root-mean-square (RMS) value for each frame**
  <br>
  Nilai RMS dipilih karena mewakili energi sinyal dan dapat memberikan indikasi intensitas suara dalam setiap frame audio.

**Selanjutnya, saya mengekstrak nilai-nilai statistik berikut dari masing-masing fitur tersebut:**

- **Mean**
  Rata-rata digunakan untuk melihat kecenderungan sentral dari distribusi data.
- **Standard Deviation**
  Mengukur seberapa tersebar data dari rata-ratanya, memberikan informasi tentang variasi dalam fitur.
- **Maximum**
  Menangkap nilai puncak dalam fitur, yang bisa menunjukkan anomali atau karakteristik penting.
- **Median**
  Memberikan gambaran distribusi data yang lebih tahan terhadap outliers dibandingkan mean.
- **Variance**
  Mengukur variasi dalam data yang lebih sensitif terhadap nilai ekstrem.
- **Skewness**
  Mengukur asimetri distribusi data, penting untuk memahami kecenderungan fitur.
- **Quartile 1**
  Menggunakan kuartil 1 untuk melihat distribusi data di sekitar median, yang membantu dalam identifikasi distribusi fitur.
- **Quartile 3**
  Menggunakan kuartil 3 untuk melihat distribusi data di sekitar median, yang membantu dalam identifikasi distribusi fitur.
- **Interquartile Range (IQR)**
  Menyediakan informasi tentang rentang distribusi utama, berguna dalam mendeteksi outliers.
- **MinMax**
  Menggunakan nilai minimum dan maksimum untuk mendapatkan rentang penuh dari data.
- **Kurtosis**
  Mengukur keterpusatan data, yang membantu dalam memahami distribusi data terhadap distribusi normal.

**Selain itu, saya juga menambahkan fitur tempo untuk masing-masing file audio.**
**Tempo ditambahkan untuk setiap file audio untuk memberikan informasi tentang kecepatan beat atau irama dalam sinyal audio, yang dapat berpengaruh pada pola dalam sinyal suara dan berguna dalam klasifikasi.**

**Terdapat penelitian yang menggunakan nilai-nilai statistik di atas, dengan menghasilkan performa model yang sangat baik. [8]**

**Dengan langkah-langkah ini, model dapat lebih mudah mengidentifikasi pola penting yang mungkin tidak terlihat dengan fitur asli dari data audio mentah. Pemilihan dan rekayasa fitur ini juga mengatasi masalah data tidak seimbang dengan lebih baik karena fitur-fitur statistik yang dipilih dapat merangkum karakteristik penting dari setiap rekaman audio tanpa memperhatikan durasinya sekaligus akan lebih mudah dilakukan oversampling.**

**Berikut adalah tampilan 5 data teratas dari data yang telah diekstraksi:**

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Mean mfcc</th>
      <th>Std Dev mfcc</th>
      <th>Max mfcc</th>
      <th>Median mfcc</th>
      <th>Variance mfcc</th>
      <th>Skewness mfcc</th>
      <th>Quartile 1 mfcc</th>
      <th>Quartile 3 mfcc</th>
      <th>IQR mfcc</th>
      <th>...</th>
      <th>Max rms</th>
      <th>Median rms</th>
      <th>Variance rms</th>
      <th>Skewness rms</th>
      <th>Quartile 1 rms</th>
      <th>Quartile 3 rms</th>
      <th>IQR rms</th>
      <th>MinMax rms</th>
      <th>Kurtosis rms</th>
      <th>Tempo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>artifact</td>
      <td>-28.484547</td>
      <td>133.570877</td>
      <td>323.493347</td>
      <td>-0.662885</td>
      <td>17841.177734</td>
      <td>-4.075856</td>
      <td>-3.428067</td>
      <td>1.804876</td>
      <td>5.232943</td>
      <td>...</td>
      <td>0.183695</td>
      <td>0.000160</td>
      <td>1.974953e-04</td>
      <td>10.701381</td>
      <td>0.000113</td>
      <td>0.000361</td>
      <td>0.000248</td>
      <td>0.183622</td>
      <td>124.567114</td>
      <td>114.843750</td>
    </tr>
    <tr>
      <th>1</th>
      <td>artifact</td>
      <td>-34.966278</td>
      <td>178.734619</td>
      <td>313.957581</td>
      <td>0.100107</td>
      <td>31946.064453</td>
      <td>-3.899768</td>
      <td>-9.097509</td>
      <td>5.817755</td>
      <td>14.915264</td>
      <td>...</td>
      <td>0.005230</td>
      <td>0.000225</td>
      <td>5.114443e-07</td>
      <td>3.970705</td>
      <td>0.000022</td>
      <td>0.000412</td>
      <td>0.000391</td>
      <td>0.005214</td>
      <td>18.582176</td>
      <td>120.185320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>artifact</td>
      <td>-31.556053</td>
      <td>163.803879</td>
      <td>302.239532</td>
      <td>-3.159485</td>
      <td>26831.712891</td>
      <td>-3.535077</td>
      <td>-14.686425</td>
      <td>6.761555</td>
      <td>21.447980</td>
      <td>...</td>
      <td>0.004233</td>
      <td>0.000227</td>
      <td>3.992640e-07</td>
      <td>2.560501</td>
      <td>0.000157</td>
      <td>0.000525</td>
      <td>0.000367</td>
      <td>0.004125</td>
      <td>7.743248</td>
      <td>123.046875</td>
    </tr>
    <tr>
      <th>3</th>
      <td>artifact</td>
      <td>-15.943065</td>
      <td>110.068741</td>
      <td>326.286743</td>
      <td>-5.171048</td>
      <td>12115.126953</td>
      <td>-2.291386</td>
      <td>-15.684322</td>
      <td>3.360450</td>
      <td>19.044772</td>
      <td>...</td>
      <td>0.103643</td>
      <td>0.010907</td>
      <td>1.083698e-04</td>
      <td>4.143348</td>
      <td>0.007867</td>
      <td>0.015477</td>
      <td>0.007610</td>
      <td>0.100495</td>
      <td>25.573779</td>
      <td>117.453835</td>
    </tr>
    <tr>
      <th>4</th>
      <td>artifact</td>
      <td>-15.301639</td>
      <td>95.154922</td>
      <td>316.893127</td>
      <td>-4.762616</td>
      <td>9054.459961</td>
      <td>-2.769833</td>
      <td>-12.728859</td>
      <td>2.792716</td>
      <td>15.521575</td>
      <td>...</td>
      <td>0.287264</td>
      <td>0.008104</td>
      <td>1.039968e-03</td>
      <td>4.539478</td>
      <td>0.006142</td>
      <td>0.011690</td>
      <td>0.005548</td>
      <td>0.286077</td>
      <td>24.663706</td>
      <td>126.048018</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>

**Saat ini kita memiliki 45 Fitur**

## Data Understanding #2
**Setelah berhasil mendapatkan data baru melalui feature engineering, langkah selanjutnya adalah melakukan preprocessing data untuk kedua kalinya namun sebelum itu kita perlu memahami datanya terlebih dahulu. Tujuan dari pemahaman data ini adalah untuk memastikan bahwa data yang telah diekstraksi siap digunakan dalam proses pelatihan model.**


- **Memeriksa Null Value dan Mengeleminasinya**
  - **Tahap ini bertujuan untuk memastikan bahwa data tidak mengandung nilai yang hilang (null values). Jika ditemukan, nilai yang hilang akan diatasi dengan cara menghapusnya atau mengisi (impute) dengan nilai yang sesuai agar tidak mempengaruhi hasil analisis dan pelatihan model.**

  **Berikut adalah Daftar Fitur yang Hilang**

  - Skewness chroma_stft      1
  - Kurtosis chroma_stft      1
  - Skewness chroma_cqt       1
  - Kurtosis chroma_cqt       1
  - Skewness rms              1
  - Kurtosis rms              1

  ```
  1 rows × 46 columns
  ```
  **Setelah dilakukan inspeksi hanya terdapat 1 data yang memiliki nilai NaN**


## Data Preprocessing #2

**Tahapan preprocessing kedua meliputi:**

- **Perlakuan terhadap data null, NaN dan sebagainya**
  <br>
  **Setelah dilakukan inspeksi, saya lebih memilih menghapusnya karena hanya 1 data saja yang memiliki nilai NaN**

- **Membagi data Label dan data Fitur kemudian Encode data Label**
  <br>
  **Pada tahap ini, dataset dibagi menjadi dua bagian, yaitu fitur dan label. Fitur adalah variabel yang digunakan untuk memprediksi label, sementara label adalah hasil yang ingin diprediksi. Jika data label berupa kategori atau string, perlu dilakukan encoding untuk mengubahnya menjadi format numerik. Encoding ini penting karena sebagian besar algoritma machine learning memerlukan data numerik sebagai input. Dengan mengubah data label ke dalam format numerik, model dapat lebih mudah memproses dan memahami informasi, yang dapat meningkatkan akurasi dan performa model. Pada label, saya menerapkan Encoder yang berupa Label Encoder.**

- **Balancing data**
  <br>
  **Sebelumnya kita tahu bahwa data mentah dengan masing-masing Label tidak seimbang, Pada tahap ini saya menerapkan teknik balancing data dengan menambah jumlah sampel pada kelas yang kurang dengan metode seperti SMOTE (Synthetic Minority Over-sampling Technique) untuk menghasilkan data sintetis.Menggunakan teknik oversampling seperti SMOTE digunakan untuk mencapai distribusi yang lebih seimbang. Dengan memastikan data yang seimbang, model akan belajar dari representasi kelas yang lebih akurat, yang membantu menghindari bias dan meningkatkan generalisasi model.**

  **Berikut adalah hasil dari oversampling balancing data:**

 
  ![balanced_data](https://github.com/user-attachments/assets/8fd6f1aa-276c-40bb-b1f0-b81dbff55651)

  **Dengan menggunakan SMOTE, data berhasil seimbang antara data lainnya**

- **Penerapan PCA**
  <br>
  **PCA dapat digunakan untuk mereduksi dimensi dari fitur ekstraksi seperti MFCCs, chroma features, dan lainnya, yang sering kali menghasilkan jumlah fitur yang sangat besar, pada kasus ini terdapat 45 fitur. Dengan menggunakan PCA, fitur-fitur tersebut dapat diringkas menjadi beberapa komponen utama yang tetap mempertahankan informasi penting, membantu model dalam proses pelatihan dan evaluasi.**

  **Saya menerapkan 30 fitur saja dari 45 fitur yang akan dipilih oleh PCA, berikut adalah fitur-fiturnya dengan urutan bobot dari yang paling tingi:**
  ```
  ['IQR chroma_cqt', 'IQR chroma_stft', 'IQR mfcc', 'IQR rms',
  'Kurtosis chroma_cqt', 'Kurtosis chroma_stft', 'Kurtosis mfcc',
  'Kurtosis rms', 'Max chroma_cqt', 'Max chroma_stft', 'Max mfcc',
  'Max rms', 'Mean chroma_cqt', 'Mean chroma_stft', 'Mean mfcc',
  'Mean rms', 'Median chroma_cqt', 'Median chroma_stft', 'Median mfcc',
  'Median rms', 'MinMax chroma_cqt', 'MinMax chroma_stft', 'MinMax mfcc',
  'MinMax rms', 'Quartile 1 chroma_cqt', 'Quartile 1 chroma_stft',
  'Quartile 1 mfcc', 'Quartile 1 rms', 'Quartile 3 chroma_cqt',
  'Quartile 3 chroma_stft']
  ```

- **Membagikan data training dan testing**
  <br>
  **Data pelatihan (training) dan data pengujian (testing). Data pelatihan digunakan untuk melatih model, sedangkan data pengujian digunakan untuk mengevaluasi kinerja model yang telah dilatih. Dengan memiliki data pengujian yang terpisah, kita dapat melakukan evaluasi kinerja yang lebih objektif dan valid, mengukur berbagai metrik seperti akurasi, sensitivitas, dan spesifisitas model.**

  **Pada penelitian kali ini, saya membagikan training sebesar 80% dan testing sebesar 20% dengan random state sebesar 42**

- **Standarisasi data yang telah terpisah**
  <br>
  **Pada tahap ini, saya menerapkan standarisasi pada data training. kemudian standard scaler tersebut diimplementasikan di data testing. Model machine learning sering kali berfungsi lebih baik ketika fitur memiliki skala yang konsisten. Standarisasi dapat mempercepat konvergensi dalam algoritma optimasi dan meningkatkan akurasi model.**

  **Berikut adalah cuplikan kode standarisasi pada data training:**

  ```
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  print(type(X_train))
  X_train = pd.DataFrame(X_train, columns=X_resampling_pca.columns)
  ```
  
  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>0</th>
        <th>1</th>
        <th>2</th>
        <th>3</th>
        <th>4</th>
        <th>5</th>
        <th>6</th>
        <th>7</th>
        <th>8</th>
        <th>9</th>
        <th>...</th>
        <th>20</th>
        <th>21</th>
        <th>22</th>
        <th>23</th>
        <th>24</th>
        <th>25</th>
        <th>26</th>
        <th>27</th>
        <th>28</th>
        <th>29</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>count</th>
        <td>1420.0000</td>
        <td>1420.0000</td>
        <td>1420.0000</td>
        <td>1420.0000</td>
        <td>1420.0000</td>
        <td>1420.0000</td>
        <td>1420.0000</td>
        <td>1420.0000</td>
        <td>1420.0000</td>
        <td>1420.0000</td>
        <td>...</td>
        <td>1420.0000</td>
        <td>1420.0000</td>
        <td>1420.0000</td>
        <td>1420.0000</td>
        <td>1420.0000</td>
        <td>1420.0000</td>
        <td>1420.0000</td>
        <td>1420.0000</td>
        <td>1420.0000</td>
        <td>1420.0000</td>
      </tr>
      <tr>
        <th>mean</th>
        <td>-0.0000</td>
        <td>-0.0000</td>
        <td>-0.0000</td>
        <td>0.0000</td>
        <td>-0.0000</td>
        <td>0.0000</td>
        <td>-0.0000</td>
        <td>-0.0000</td>
        <td>0.0000</td>
        <td>-0.0000</td>
        <td>...</td>
        <td>-0.0000</td>
        <td>-0.0000</td>
        <td>0.0000</td>
        <td>0.0000</td>
        <td>-0.0000</td>
        <td>0.0000</td>
        <td>0.0000</td>
        <td>0.0000</td>
        <td>-0.0000</td>
        <td>0.0000</td>
      </tr>
      <tr>
        <th>std</th>
        <td>1.0004</td>
        <td>1.0004</td>
        <td>1.0004</td>
        <td>1.0004</td>
        <td>1.0004</td>
        <td>1.0004</td>
        <td>1.0004</td>
        <td>1.0004</td>
        <td>1.0004</td>
        <td>1.0004</td>
        <td>...</td>
        <td>1.0004</td>
        <td>1.0004</td>
        <td>1.0004</td>
        <td>1.0004</td>
        <td>1.0004</td>
        <td>1.0004</td>
        <td>1.0004</td>
        <td>1.0004</td>
        <td>1.0004</td>
        <td>1.0004</td>
      </tr>
      <tr>
        <th>min</th>
        <td>-1.5564</td>
        <td>-2.7517</td>
        <td>-3.4097</td>
        <td>-4.2118</td>
        <td>-2.0560</td>
        <td>-6.9714</td>
        <td>-4.0883</td>
        <td>-8.7877</td>
        <td>-6.3543</td>
        <td>-13.1773</td>
        <td>...</td>
        <td>-5.3258</td>
        <td>-6.9095</td>
        <td>-5.6671</td>
        <td>-6.5501</td>
        <td>-3.6511</td>
        <td>-4.9615</td>
        <td>-8.7210</td>
        <td>-7.5996</td>
        <td>-6.6504</td>
        <td>-6.6312</td>
      </tr>
      <tr>
        <th>25%</th>
        <td>-0.6684</td>
        <td>-0.5511</td>
        <td>-0.4258</td>
        <td>-0.4597</td>
        <td>-0.4389</td>
        <td>-0.3323</td>
        <td>-0.5325</td>
        <td>-0.4932</td>
        <td>-0.5515</td>
        <td>-0.3667</td>
        <td>...</td>
        <td>-0.4448</td>
        <td>-0.3391</td>
        <td>-0.6473</td>
        <td>-0.5420</td>
        <td>-0.4506</td>
        <td>-0.4829</td>
        <td>-0.3341</td>
        <td>-0.4382</td>
        <td>-0.4438</td>
        <td>-0.3590</td>
      </tr>
      <tr>
        <th>50%</th>
        <td>-0.2229</td>
        <td>-0.2304</td>
        <td>0.1671</td>
        <td>0.0245</td>
        <td>-0.1583</td>
        <td>0.1289</td>
        <td>-0.1687</td>
        <td>0.0429</td>
        <td>0.0424</td>
        <td>0.0135</td>
        <td>...</td>
        <td>-0.0167</td>
        <td>-0.0306</td>
        <td>0.0257</td>
        <td>0.0161</td>
        <td>0.0152</td>
        <td>0.0165</td>
        <td>-0.0071</td>
        <td>-0.0455</td>
        <td>-0.0084</td>
        <td>-0.0270</td>
      </tr>
      <tr>
        <th>75%</th>
        <td>0.4321</td>
        <td>0.3393</td>
        <td>0.5974</td>
        <td>0.5344</td>
        <td>0.2730</td>
        <td>0.5293</td>
        <td>0.3544</td>
        <td>0.6134</td>
        <td>0.7519</td>
        <td>0.4173</td>
        <td>...</td>
        <td>0.5252</td>
        <td>0.3083</td>
        <td>0.6949</td>
        <td>0.5768</td>
        <td>0.4253</td>
        <td>0.4977</td>
        <td>0.3748</td>
        <td>0.4205</td>
        <td>0.4950</td>
        <td>0.3223</td>
      </tr>
      <tr>
        <th>max</th>
        <td>5.7424</td>
        <td>3.7433</td>
        <td>3.1430</td>
        <td>4.8601</td>
        <td>13.4303</td>
        <td>2.8787</td>
        <td>6.9124</td>
        <td>2.8134</td>
        <td>3.8830</td>
        <td>3.7511</td>
        <td>...</td>
        <td>5.1139</td>
        <td>8.3163</td>
        <td>3.6197</td>
        <td>5.6940</td>
        <td>11.2972</td>
        <td>7.4277</td>
        <td>7.4117</td>
        <td>4.9920</td>
        <td>5.6827</td>
        <td>13.6669</td>
      </tr>
    </tbody>
  </table>
  <p>8 rows × 30 columns</p>
  </div>

  **Standarisasi bertujuan untuk mengubah fitur sehingga memiliki mean 0 dan deviasi standar 1 seperti yang terlihat pada tabel di atas. Hal ini membantu dalam membuat fitur memiliki skala yang seragam, yang penting untuk algoritma machine learning yang sensitif terhadap skala data.**

  **Standarisasi data testing menggunakan scaler yang telah di-fit pada data training untuk memastikan bahwa kedua dataset berada pada skala yang seragam.**

  **Berikut adalah cuplikan kode untuk standarisasi pada data testing:**

  ```
  X_test = scaler.transform(X_test)
  print(type(X_test))
  X_test = pd.DataFrame(X_test, columns=X_resampling_pca.columns)
  ```

  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right;">
        <th></th>
        <th>0</th>
        <th>1</th>
        <th>2</th>
        <th>3</th>
        <th>4</th>
        <th>5</th>
        <th>6</th>
        <th>7</th>
        <th>8</th>
        <th>9</th>
        <th>...</th>
        <th>20</th>
        <th>21</th>
        <th>22</th>
        <th>23</th>
        <th>24</th>
        <th>25</th>
        <th>26</th>
        <th>27</th>
        <th>28</th>
        <th>29</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>count</th>
        <td>355.0000</td>
        <td>355.0000</td>
        <td>355.0000</td>
        <td>355.0000</td>
        <td>355.0000</td>
        <td>355.0000</td>
        <td>355.0000</td>
        <td>355.0000</td>
        <td>355.0000</td>
        <td>355.0000</td>
        <td>...</td>
        <td>355.0000</td>
        <td>355.0000</td>
        <td>355.0000</td>
        <td>355.0000</td>
        <td>355.0000</td>
        <td>355.0000</td>
        <td>355.0000</td>
        <td>355.0000</td>
        <td>355.0000</td>
        <td>355.0000</td>
      </tr>
      <tr>
        <th>mean</th>
        <td>-0.0309</td>
        <td>-0.1230</td>
        <td>0.0247</td>
        <td>-0.0304</td>
        <td>-0.0271</td>
        <td>-0.0027</td>
        <td>-0.0671</td>
        <td>-0.0342</td>
        <td>-0.0466</td>
        <td>0.0073</td>
        <td>...</td>
        <td>0.0854</td>
        <td>0.0675</td>
        <td>-0.0224</td>
        <td>0.0625</td>
        <td>-0.0783</td>
        <td>-0.0517</td>
        <td>-0.0150</td>
        <td>-0.0142</td>
        <td>0.0702</td>
        <td>-0.0602</td>
      </tr>
      <tr>
        <th>std</th>
        <td>1.0098</td>
        <td>0.8888</td>
        <td>0.9644</td>
        <td>0.9373</td>
        <td>1.0667</td>
        <td>0.9081</td>
        <td>0.8818</td>
        <td>0.9268</td>
        <td>0.9839</td>
        <td>0.7833</td>
        <td>...</td>
        <td>0.9955</td>
        <td>0.9362</td>
        <td>1.0276</td>
        <td>1.1354</td>
        <td>0.9809</td>
        <td>1.1148</td>
        <td>1.0433</td>
        <td>1.0912</td>
        <td>1.0789</td>
        <td>0.9254</td>
      </tr>
      <tr>
        <th>min</th>
        <td>-1.5502</td>
        <td>-2.3496</td>
        <td>-2.9915</td>
        <td>-3.8563</td>
        <td>-1.6272</td>
        <td>-6.0959</td>
        <td>-2.7380</td>
        <td>-5.3391</td>
        <td>-6.2095</td>
        <td>-3.7593</td>
        <td>...</td>
        <td>-3.1139</td>
        <td>-5.0804</td>
        <td>-5.3555</td>
        <td>-4.5604</td>
        <td>-3.9143</td>
        <td>-4.9613</td>
        <td>-4.2973</td>
        <td>-5.9361</td>
        <td>-4.9100</td>
        <td>-6.2577</td>
      </tr>
      <tr>
        <th>25%</th>
        <td>-0.7215</td>
        <td>-0.5835</td>
        <td>-0.4483</td>
        <td>-0.5280</td>
        <td>-0.4825</td>
        <td>-0.3039</td>
        <td>-0.5497</td>
        <td>-0.4855</td>
        <td>-0.5595</td>
        <td>-0.4134</td>
        <td>...</td>
        <td>-0.4078</td>
        <td>-0.3030</td>
        <td>-0.6252</td>
        <td>-0.5634</td>
        <td>-0.5018</td>
        <td>-0.4436</td>
        <td>-0.3873</td>
        <td>-0.4615</td>
        <td>-0.4250</td>
        <td>-0.3818</td>
      </tr>
      <tr>
        <th>50%</th>
        <td>-0.2658</td>
        <td>-0.3146</td>
        <td>0.0728</td>
        <td>0.0897</td>
        <td>-0.1741</td>
        <td>0.1166</td>
        <td>-0.2003</td>
        <td>-0.0371</td>
        <td>-0.0519</td>
        <td>-0.0322</td>
        <td>...</td>
        <td>0.0430</td>
        <td>0.0174</td>
        <td>0.0088</td>
        <td>0.0684</td>
        <td>-0.0435</td>
        <td>-0.0876</td>
        <td>-0.0604</td>
        <td>-0.0337</td>
        <td>0.0082</td>
        <td>-0.0104</td>
      </tr>
      <tr>
        <th>75%</th>
        <td>0.3762</td>
        <td>0.1807</td>
        <td>0.5875</td>
        <td>0.5307</td>
        <td>0.1878</td>
        <td>0.4463</td>
        <td>0.3257</td>
        <td>0.6059</td>
        <td>0.7080</td>
        <td>0.3490</td>
        <td>...</td>
        <td>0.5578</td>
        <td>0.3732</td>
        <td>0.6924</td>
        <td>0.6375</td>
        <td>0.3648</td>
        <td>0.3979</td>
        <td>0.3544</td>
        <td>0.5032</td>
        <td>0.5043</td>
        <td>0.2955</td>
      </tr>
      <tr>
        <th>max</th>
        <td>5.5641</td>
        <td>2.9636</td>
        <td>3.0770</td>
        <td>2.5269</td>
        <td>9.1703</td>
        <td>2.8447</td>
        <td>3.2418</td>
        <td>2.3058</td>
        <td>2.1162</td>
        <td>3.8239</td>
        <td>...</td>
        <td>4.3931</td>
        <td>6.5264</td>
        <td>2.8094</td>
        <td>7.8304</td>
        <td>6.0926</td>
        <td>7.0636</td>
        <td>6.5862</td>
        <td>4.0259</td>
        <td>9.6041</td>
        <td>5.6410</td>
      </tr>
    </tbody>
  </table>
  <p>8 rows × 30 columns</p>
  </div>

  **Dari tabel di atas terlihat bahwa mean tidak 0 seperti sebelumnya dan standar deviasi juga tidak 1, hal ini tidak masalah dan menunjukkan bahwa tidak menimbulkan kebocoran data. Informasi tentang data uji (yang seharusnya tidak dilihat oleh model) turut diikutsertakan dalam proses transformasi data latih. Oleh karena itu, proses standarisasi dilakukan setelah membagi data training dan testing.**

## Modeling

**Pada tahap ini saya membandingkan dua algoritma klasifikasi, yaitu Machine Learning Random Forest dan Deep Learning LSTM**

**Random Forest:**
<br>

**Random Forest adalah algoritma berbasis ensemble learning yang menggabungkan prediksi dari banyak pohon keputusan (decision trees) untuk meningkatkan akurasi prediksi dan mengurangi risiko overfitting. Setiap pohon keputusan dihasilkan dari subset acak dari data pelatihan, dan hasil akhirnya ditentukan melalui voting mayoritas dari semua pohon. Pada data saya, Random Forest bekerja dengan membangun banyak pohon keputusan menggunakan fitur-fitur statistik yang diekstraksi dari data audio (data tabular). Setiap pohon belajar dari subset data yang berbeda, dan output dari semua pohon dikombinasikan untuk menghasilkan keputusan akhir. Proses ini membuat Random Forest sangat baik dalam menangani variasi dalam data dan mengurangi kemungkinan overfitting.**

- **Kelebihan:**
  - **Robust terhadap Overfitting:** Karena menggunakan agregasi dari banyak pohon keputusan, Random Forest cenderung tidak overfit pada data pelatihan.
  - **Tidak memerlukan banyak tuning parameter:** Sehingga mudah dan tidak memerlukan banyak waktu untuk tuning algoritma Random Forest
- **Kekurangan:**
  - **Ukuran model yang besar:** Meskipun tidak cenderung overfit, model Random Forest memiliki ukuran file yang sangat besar
  - **Kurang Efektif untuk Data dengan Dimensi Tinggi:** Untuk dataset dengan banyak fitur, Random Forest mungkin tidak bekerja sebaik metode lain yang dapat mengelola dimensi tinggi dengan lebih baik.

**LSTM:**
<br>

**LSTM adalah jenis jaringan saraf tiruan (neural network) yang dirancang untuk mengenali pola dalam data sekuensial. LSTM dapat digunakan pada data tabular. Dalam konteks data tabular, LSTM memproses setiap fitur dalam urutan tertentu, memungkinkan model untuk menangkap hubungan jangka panjang antar fitur yang mungkin tidak ditangkap oleh algoritma tradisional.**

- **Kelebihan:**
  - **Memori Jangka Panjang:** LSTM memiliki kemampuan untuk menyimpan informasi jangka panjang yang berguna untuk memahami pola dalam data.
  - **Fleksibilitas dan Kinerja Tinggi:** Dapat mencapai kinerja tinggi pada berbagai jenis masalah.
- **Kekurangan:**
  - **Kompleksitas Model dan Latihan:** LSTM adalah model yang lebih kompleks dibandingkan dengan algoritma machine learning tradisional, yang dapat membuat pelatihan memakan waktu dan memerlukan lebih banyak sumber daya komputasi.
  - **Tuning Parameter yang Kompleks:** Menyempurnakan hyperparameter LSTM bisa menjadi proses yang rumit dan memerlukan eksperimen ekstensif untuk menemukan kombinasi yang optimal.

**Terdapat beberapa langkah untuk mengimplementasikan masing-masing model algoritma, antara lain:**

- **Implementasi Grid Search untuk Hyperparameter Tuning:** Menggunakan GridSearch untuk mengeksplorasi berbagai kombinasi hyperparameter model secara sistematis, sehingga dapat menemukan konfigurasi yang memberikan performa terbaik berdasarkan metrik evaluasi yang ditentukan yaitu akurasi.
- **Implementasi K-Fold Cross Validation:** Setelah berhasil mendapatkan parameter yang terbaik, saya mengimplementasikan K-Fold Cross Validation sebanyak 7 dengan parameter tersebut. Terdapat penelitian yang membuktikan 7 fold memberikan performa lebih baik daripada yang lainnya. [4]
- **Menguji robustness model dengan data yang belum dilihatnya:** Setelah model melakukan pelatihan, tahap selanjutnya yaitu menguji ketahanan semua model dari fold dengan data yang belum dilihatnya. Di tahap inilah model akan diseleksi untuk dibandingkan dengan algoritma lainnya.


### Random Forest

- **Implementasi Grid Search untuk Hyperparameter Tuning**
  **Dengan menggunakan data training, dilakukan hyperparameter tuning dengan parameter sebagai berikut:**
    - 'n_estimators': [100, 200]
      <br> Ini adalah jumlah pohon keputusan yang akan dibuat dalam model Random Forest. Semakin banyak pohon, semakin baik model dalam menangkap pola dari data, tetapi juga semakin tinggi waktu komputasi. 
    - 'max_depth': [16, 32, 64, 128, 256]
      <br> Menentukan kedalaman maksimum dari setiap pohon. Semakin dalam pohon, semakin kompleks model, yang dapat meningkatkan akurasi tetapi juga meningkatkan risiko overfitting.
    - 'min_samples_split': [2, 5, 10]
      <br> Jumlah minimum sampel yang diperlukan untuk membagi node. Nilai yang lebih tinggi mencegah model menjadi terlalu kompleks dengan memaksa pembagian hanya ketika ada sejumlah sampel yang signifikan.
    - 'min_samples_leaf': [1, 2, 4]
      <br> Jumlah minimum sampel yang diperlukan di node daun (leaf node). Nilai yang lebih tinggi menghasilkan pohon yang lebih seimbang dan mengurangi risiko overfitting.
    <br>

- **Implementasi K-Fold Cross Validation**
  **Menggunakan parameter yang telah didapatkan dari Hyperparameter Tuning, saya mengimplementasikan K-Fold sebanyak 7**

- **Menguji robustness semua model dengan data yang belum dilihatnya**
  

### LSTM

- **Implementasi Grid Search untuk Hyperparameter Tuning**
  **Dengan menggunakan data training, dilakukan hyperparameter tuning dengan parameter sebagai berikut:**
  - 'batch_size': [64, 128, 256]
    <br> Ukuran batch menentukan jumlah sampel yang diproses sebelum model diperbarui. Batch size yang lebih kecil cenderung membuat model lebih sensitif terhadap gradien, yang dapat membantu mencapai konvergensi yang lebih cepat.
  - 'epochs': [50, 100, 150, 200]
    <br> Jumlah epoch adalah berapa kali seluruh dataset melewati jaringan selama pelatihan. Jumlah yang lebih tinggi biasanya berarti model akan dilatih lebih lama dan mungkin mencapai hasil yang lebih baik, tetapi juga meningkatkan risiko overfitting. 
  <br>
     

-  **Implementasi K-Fold Cross Validation**
  **Menggunakan parameter yang telah didapatkan dari Hyperparameter Tuning, saya mengimplementasikan K-Fold sebanyak 7**

- **Menguji robustness model dengan data yang belum dilihatnya**

## Evaluation

### Random Forest

- **Hasil dari Parameter Tuning, memberikan nilai terbaik sebesar 90.92% akurasi dengan parameter 'max_depth': 32, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200**

- **Dengan mengimplementasikan 7-Fold Cross Validation, hasil yang terbaik yaitu Fold ke-3 dengan skor pada data training sebagai berikut:**
  - Akurasi sebesar 96.06%
  - Sensitivitas 96% 
  - Spesifisitas 95.89%
  - F1-Score 95.93%
  <br>
  
  **Berikut adalah rata-rata untuk masing-masing metrik:**
  <br>
  
    - Rata-rata Akurasi: 90.91%
    - Rata-rata Sensitivitas: 90.68%
    - Rata-rata Spesifisitas: 90.61%
    - Rata-rata F1 Score: 90.21%

- **Dengan menguji semua model, Fold pertama meraih nilai performa paling tinggi yaitu:**

  - Accuracy:  91.55%
  - Sensitivity:  92.67%
  - Precision:  91.58%
  - F1 Score:  91.86%
  <br>
  
  **Berikut adalah rata-rata dari setiap metrik dalam fold:**
    - **Accuracy:** 91.05%
    - **Sensitivity:** 92.01%
    - **Precision:** 92.10%
    - **F1 Score:** 92.00%

**Metric model rata-rata inilah yang akan dipilih sebagai perbandingan dengan algoritma model lain**

### LSTM

- **Hasil dari Parameter Tuning, memberikan nilai terbaik sebesar 81.9% akurasi dengan parameter 'batch_size': 64, 'epochs': 200**

**Dengan mengimplementasikan 7-Fold Cross Validation, hasil yang terbaik yaitu Fold ke-3 dengan skor pada data training sebagai berikut:**
  - Akurasi 85.71%
  - Sensitivitas 84.94% 
  - Spesifitas 85.06%
  - F1 Score 84.87%
  <br>
  
  **Berikut adalah rata-rata untuk masing-masing metrik:**
  <br>
  - Akurasi: 79.64%
  - Sensitivitas: 79.31%
  - Spesifisitas: 78.89%
  - F1 Score: 78.39%

- **Dengan menguji semua model, fold pertama meraih nilai performa paling tinggi yaitu:**

  - Accuracy:  82.82%
  - Sensitivity:  84.33%
  - Precision:  83.05%
  - F1 Score:  83.38%
  <br>
  
  **Berikut adalah rata-rata dari setiap metrik dalam fold:**
  <br>
  
  - **Akurasi:** 76.74%
  - **Sensitivitas:** 78.93%
  - **Presisi:** 77.14%
  - **F1 Score:** 77.07%

**Metric model rata-rata inilah yang akan dipilih sebagai perbandingan dengan algoritma model lain**

### Perbandingan Model terbaik dari masing-masing algoritma pada data yang belum mereka lihat

<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Random Forest</th>
      <th>LSTM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Accuracy</strong></td>
      <td>91.05%</td>
      <td>76.74%</td>
    </tr>
    <tr>
      <td><strong>Sensitivity</strong></td>
      <td>92.01%</td>
      <td>78.93%</td>
    </tr>
    <tr>
      <td><strong>Precision</strong></td>
      <td>92.10%</td>
      <td>77.14%</td>
    </tr>
    <tr>
      <td><strong>F1 Score</strong></td>
      <td>92.00%</td>
      <td>77.07%</td>
    </tr>
  </tbody>
</table>


**Berdasarkan hasil metrik kinerja di atas, Random Forest adalah model terbaik untuk digunakan dalam kasus ini karena menunjukkan kinerja yang lebih unggul di semua metrik evaluasi.**

**Berdasarkan studi, metric yang digunakan pada kasus diagnosis medis yaitu sensitivitas. Mengukur dan membandingkan sensitivitas berbagai tes atau model untuk menentukan seberapa efektif mereka dalam mendeteksi penyakit atau kondisi tertentu. Sensitivitas mengukur proporsi kasus positif yang benar.**

**Sensitivitas (Recall):**

True Positives (TP) / (True Positives (TP) + False Negatives (FN))

**Tes dengan sensitivitas tinggi akan menghasilkan sedikit False Negatives (FN), yaitu kasus di mana pasien yang sakit malah didiagnosis negatif oleh tes. Dengan kata lain, jika tes sangat sensitif, hampir semua orang yang sakit akan terdeteksi oleh tes tersebut. Sensitivitas tinggi sangat penting ketika risiko gagal mendeteksi penyakit bisa berakibat serius, seperti pada penyakit yang mengancam jiwa. [5]**

**Model yang dipilih yaitu Random Forest memiliki Sensitivitas yang tinggi yaitu 92.67%**

**Dari perspektif Business Understanding, keberhasilan model ini dalam mencapai sensitivitas yang tinggi langsung berdampak pada efektivitas layanan medis yang disediakan. Penggunaan model dengan sensitivitas tinggi memastikan bahwa penyakit terdeteksi lebih awal dan lebih akurat, yang pada gilirannya meningkatkan tingkat keberhasilan perawatan, mengurangi biaya pengobatan jangka panjang, dan meningkatkan kepuasan pasien. Selain itu, denosing memiliki nilai yang sangat dalam membersihkan data audio.**

**Dengan demikian, solusi yang direncanakan dalam penelitian ini tidak hanya terbukti berhasil secara teknis, tetapi juga berdampak signifikan pada tujuan bisnis, yaitu meningkatkan akurasi dan kecepatan dalam diagnosis medis, yang berujung pada peningkatan kualitas perawatan kesehatan.**

**Tujuan penelitian ini telah berhasil dicapai, dan problem statement sudah terpenuhi. Peningkatan performa model dengan sensitivitas di atas 90% dan efisiensi denoising model sebesar 0.92 STOI semua ini membuktikan bahwa model Random Forest adalah pilihan yang tepat untuk mendukung kebutuhan diagnostik yang cepat dan akurat.**

## Referensi
[1] [Cardiovascular Disease](https://www.who.int/health-topics/cardiovascular-diseases#tab=tab_1)
<br>

[2] [C. H. Taal, R. C. Hendriks, R. Heusdens and J. Jensen, "A short-time objective intelligibility measure for time-frequency weighted noisy speech," 2010 IEEE International Conference on Acoustics, Speech and Signal Processing, Dallas, TX, USA, 2010, pp. 4214-4217](10.1109/ICASSP.2010.5495701)
<br>

[3] [Turky N. Alotaiby, Saud Rashid Alrshoud, Saleh A. Alshebeili, Latifah M. Aljafar, ”ECG-based subject identification using statistical features and random forest”, Journal of Sensors, vol. 2019, Article ID 6751932, 13 pages, 2019.](doi.org/10.1155/2019/6751932)
<br>

[4] [Nti, Isaac & Nyarko-Boateng, Owusu & Aning, Justice. (2021). Performance of Machine Learning Algorithms with Different K Values in K-fold Cross-Validation. International Journal of Information Technology and Computer Science. 6. 61-71. 10.5815/ijitcs.2021.06.05.](http://dx.doi.org/10.5815/ijitcs.2021.06.05)
<br>

[5] [Shreffler J, Huecker MR. Diagnostic Testing Accuracy: Sensitivity, Specificity, Predictive Values and Likelihood Ratios.](https://www.ncbi.nlm.nih.gov/books/NBK557491/)
<br>

[6] [Raza, A.; Mehmood, A.; Ullah, S.; Ahmad, M.; Choi, G.S.; On, B.-W. Heartbeat Sound Signal Classification Using Deep Learning. Sensors 2019, 19, 4819. https://doi.org/10.3390/s19214819](https://doi.org/10.3390/s19214819)
<br>

[7] [Bentley, P.; Nordehn, G.; Coimbra, M.; Mannor, S. The PASCAL Classifying Heart Sounds Challenge 2011 (CHSC2011) Results. Available online: http://www.peterjbentley.com/heartchallenge/index.html](http://www.peterjbentley.com/heartchallenge/index.html)
<br>

[8] [Nia Madu Marliana, Satria Mandala, Hau, Y. W., & Yafooz, W. M. (2023). Multiclass Classification of Myocardial Infarction Based on Phonocardiogram Signals Using Ensemble Learning. Jurnal Nasional Teknik Elektro, 12(3), 7–12. https://doi.org/10.25077/jnte.v12n3.1121.2023](https://doi.org/10.25077/jnte.v12n3.1121.2023)
<br>

[9] [Liu, S., Sabrina, N., & Hardson, H. (2023). Comparison of FIR and IIR Filters for Audio Signal Noise Reduction. Ultima Computing : Jurnal Sistem Komputer, 15(1), 19-24.](https://doi.org/10.31937/sk.v15i1.3171)

**---Ini adalah bagian akhir laporan---**
