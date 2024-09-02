# %%
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Audio
import scipy.signal as signal
import numpy as np
from pystoi import stoi
import scipy.stats as stats
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, models
from scikeras.wrappers import KerasClassifier 

# %% [markdown]
# # Data Loading

# %%
DATASET_URL = 'https://www.kaggle.com/datasets/mersico/dangerous-heartbeat-dataset-dhd'

# %%
#import dataset as dataframe
data_labels = pd.read_csv('DHD/labels.csv')
metadata = pd.read_csv('DHD/metadata.csv')

# %% [markdown]
# # Exploratory Data Analysis

# %%
data_labels.head()

# %%
metadata.head()

# %%
metadata.info()

# %%
label_count = data_labels['label'].value_counts()
label_percentages = label_count / label_count.sum() * 100

# %%
# Bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=label_count.index, y=label_count.values, palette='viridis')
plt.xlabel('Label')
plt.ylabel('Jumlah')
plt.title('Bar Plot dari Label')
plt.show()

# Pie chart
plt.figure(figsize=(8, 8))
plt.pie(label_percentages, labels=label_percentages.index, autopct='%1.2f%%', colors=sns.color_palette('viridis', len(label_percentages)))
plt.title('Pie Chart dari Label')
plt.show()

# %% [markdown]
# Label yang ada pada data penyakit jantung ini terdiri dari Normal, murmur, artifact, extrastole, dan extrahls. Namun, data yang diterima tidak seimbang terlihat Normal lebih dominan.

# %%
metadata.describe()

# %% [markdown]
# Terlihat adanya ketidakseragaman pada fitur sample_rate dan num_channels

# %%
sample_rate_count = metadata['sample_rate'].value_counts()
sample_rate_percentages = sample_rate_count / sample_rate_count.sum() * 100

# %%
# Membuat bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=sample_rate_count.index, y=sample_rate_count.values, palette='viridis')
plt.xlabel('Sample Rate')
plt.ylabel('Jumlah')
plt.title('Bar Plot dari Sample Rate')
plt.show()

# Membuat pie chart
plt.figure(figsize=(8, 8))
plt.pie(sample_rate_percentages, labels=sample_rate_percentages.index, autopct='%1.2f%%', colors=sns.color_palette('viridis', len(sample_rate_percentages)))
plt.title('Pie Chart dari Sample Rate')
plt.show()

# %% [markdown]
# Terdapat beberapa jenis sample rate pada data audio, kita akan mengonversi semuanya ke 44100 agar agar ekstraksi fitur yang kita lakukan memiliki informasi fitur sample rate yang sama

# %%
channel_count = metadata['num_channels'].value_counts()
channel_percentages = channel_count / channel_count.sum() * 100

# %%
# Membuat bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=channel_count.index, y=channel_count.values, palette='viridis')
plt.xlabel('Channel')
plt.ylabel('Jumlah')
plt.title('Bar Plot dari Channel')
plt.show()

# Membuat pie chart
plt.figure(figsize=(8, 8))
plt.pie(channel_percentages, labels=channel_percentages.index, autopct='%1.2f%%', colors=sns.color_palette('viridis', len(channel_percentages)))
plt.title('Pie Chart dari Channel')
plt.show()

# %% [markdown]
# Ditemukan data Wav dengan channel stereo (2) sementara kita hanya memerlukan data mono saja (1) kita akan mengonversi data wav tersebut ke mono agar ekstraksi fitur yang kita lakukan memiliki informasi fitur yang sama yaitu mono

# %% [markdown]
# # Data Preprocessing #1

# %% [markdown]
# ## Data Preparation

# %%
#convert semua sample rate ke 44100 Hz
for i in metadata['filename']:
    audio_file = f'DHD/audio/{i}'
    y, sr = librosa.load(audio_file, sr=None)
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=44100)
    sf.write(f'DHD/resample_audio/{i[:-4]}_resampled.wav', y_resampled, 44100)

# %%
sampling_rates_resampled = []

for i in metadata['filename']:
    data = i[:-4]
    y, sr = librosa.load(f'DHD/resample_audio/{data}_resampled.wav', sr=None)
    sampling_rates_resampled.append(sr)

sampling_rates_df = pd.DataFrame(sampling_rates_resampled, columns=['Sampling Rate'])

sampling_rates_count = sampling_rates_df['Sampling Rate'].value_counts()
sampling_rate_percentages = sampling_rates_count / sampling_rates_count.sum() * 100

# %%
# Pie chart
plt.figure(figsize=(8, 8))
plt.pie(sampling_rate_percentages, labels=sampling_rate_percentages.index, autopct='%1.2f%%', colors=sns.color_palette('viridis', len(sampling_rate_percentages)))
plt.title('Pie Chart dari Sampling Rate')
plt.show()

# %% [markdown]
# Dapat dilihat bahwa semua sample rate telah berhasil dikonversi ke 44100 Hz

# %%
#proses konversi channel
for i in range(len(metadata['filename'])):
    data = metadata['filename'][i][:-4]
    filename = f"DHD/resample_audio/{data}_resampled.wav"
    if metadata['num_channels'][i] == 2:
        y, sr = librosa.load(filename, sr=None, mono=False)
        y_mono = librosa.to_mono(y)
        sf.write(f'DHD/resample_audio_mono/{data}_mono.wav', y_mono, sr)
    else:
        y, sr = librosa.load(filename, sr=None)
        sf.write(f'DHD/resample_audio_mono/{data}_mono.wav', y, sr)

# %%
channel_converted = []

for i in range(len(metadata['filename'])):
    data = metadata['filename'][i][:-4]
    y, sr = librosa.load(f'DHD/resample_audio_mono/{data}_mono.wav', sr=None)
    channel_converted.append(y.ndim)

channel_converted_df = pd.DataFrame(channel_converted, columns=['Channel'])

channel_converted_count = channel_converted_df['Channel'].value_counts()
channel_converted_percentage = channel_converted_count / channel_converted_count.sum() * 100

# %%
# Pie chart
plt.figure(figsize=(8, 8))
plt.pie(channel_converted_percentage, labels=channel_converted_percentage.index, autopct='%1.2f%%', colors=sns.color_palette('viridis', len(channel_converted_percentage)))
plt.title('Pie Chart dari Channel')
plt.show()

# %% [markdown]
# Dapat dilihat bahwa semua channel telah berhasil dikonversi ke Mono

# %% [markdown]
# ## Denoising

# %%
# Audio Denosing menggunakan FIR
def audio_filter_FIR(audio_file, output_file):
    data, sample_rate = sf.read(audio_file)

    # Parameters untuk FIR filter
    cutoff_freq = 4000.0
    filter_order = 100

    # Normalisasi cutoff frequency
    nyquist_rate = sample_rate / 2.0
    normalized_cutoff = cutoff_freq / nyquist_rate

    # FIR filter with Hamming window
    fir_coefficients = signal.firwin(filter_order, normalized_cutoff, window='hamming')

    filtered_data = signal.lfilter(fir_coefficients, 1.0, data)
    sf.write(output_file, filtered_data, sample_rate)

# %%
# proses denoising audio
for i in range(len(metadata['filename'])):
    data = metadata['filename'][i][:-4]
    audio_filter_FIR(f'DHD/resample_audio_mono/{data}_mono.wav', f'DHD/final_audio/{data}_filtered.wav')

# %%
stoi_values = []

# proses pengukuran performa denoising
for i in range(len(metadata['filename'])):
    data = metadata['filename'][i][:-4]

    # Load the WAV files
    clean_signal, _ = sf.read(f'DHD/resample_audio_mono/{data}_mono.wav')
    denoised_signal, _ = sf.read(f'DHD/final_audio/{data}_filtered.wav')
    sample_rate = 44100
    
    # Komputasi STOI
    stoi_value = stoi(clean_signal, denoised_signal, sample_rate, extended=False)
    stoi_values.append(stoi_value)


# %%
stoi_values = np.array(stoi_values)

# %%
stoi_average = np.average(stoi_values)

print(stoi_average)

# %% [markdown]
# Skor STOI hampir sempurna yaitu 0.92 dari 1, menunjukkan performa denoising yang sangat baik

# %% [markdown]
# ### Comparation Between Raw Audio and Denoised Audio

# %%
raw_audio, sample_rate = sf.read("DHD/resample_audio_mono/artifact_2023_4_mono.wav")
denoised_audio, sample_rate = sf.read("DHD/final_audio/artifact_2023_4_filtered.wav")

min_len = min(len(raw_audio), len(denoised_audio))
raw_audio = raw_audio[:min_len]
denoised_audio = denoised_audio[:min_len]

# %%
time = np.arange(0, min_len) / sample_rate

plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(time, raw_audio, label='Raw Audio')
plt.title('Raw Audio')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(time, denoised_audio, label='Denoised Audio', color='red')
plt.title('Denoised Audio')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()


# %% [markdown]
# Mungkin terlihat bahwa raw audio dan denoised audio tidak berbeda, namun perhatikan pada detik ke-10 perbedaan begitu terlihat. Data yang digunakan merupakan data akademis yang sudah bersih, namun tetap dilakukan denoising untuk meningkatkan performa model yang akan dibuat dan simulasi pada data raw yang akan memiliki noise saat deployment. Hal ini bisa dibuktikan dengan pemutaran audio pada Audio Raw dan yang sudah di-denoising

# %%
#Raw Data
play_wav = "DHD/resample_audio_mono/artifact_2023_4_mono.wav"
Audio(play_wav)

# %%
#Denoised Data
play_wav = "DHD/final_audio/artifact_2023_4_filtered.wav"
Audio(play_wav)

# %% [markdown]
# Terdengar data yang belum di-denoising pun suaranya sudah bersih, kita lanjutkan pada sampel berikutnya

# %%
raw_audio, sample_rate = sf.read("DHD/resample_audio_mono/aortic_regurgitation_2023_3_mono.wav")
denoised_audio, sample_rate = sf.read("DHD/final_audio/aortic_regurgitation_2023_3_filtered.wav")

min_len = min(len(raw_audio), len(denoised_audio))
raw_audio = raw_audio[:min_len]
denoised_audio = denoised_audio[:min_len]

# %%
time = np.arange(0, min_len) / sample_rate

plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(time, raw_audio, label='Raw Audio')
plt.title('Raw Audio')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(time, denoised_audio, label='Denoised Audio', color='red')
plt.title('Denoised Audio')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# %% [markdown]
# Pada data sampel kali ini pun data denoised dan data raw terlihat tidak jauh berbeda, hal ini dibuktikan dari data raw yang memang sudah bersih

# %%
#Raw Data
play_wav = "DHD/resample_audio_mono/aortic_regurgitation_2023_3_mono.wav"
Audio(play_wav)

# %%
#Denoised Data
play_wav = "DHD/final_audio/aortic_regurgitation_2023_3_filtered.wav"
Audio(play_wav)

# %%
#memperbarui metadata label menjadi file yang telah didenoising
data_labels['filename'] = data_labels['filename'].str.replace('.wav', '_filtered.wav')
data_labels.to_csv('DHD/new_labels.csv', index=False)

# %% [markdown]
# # Feature Engineering

# %%
new_data_labels = pd.read_csv('DHD/new_labels.csv')

# %%
new_data_labels.head()

# %%
def compute_mfcc_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    mean_mfcc = np.mean(mfcc)
    std_mfcc = np.std(mfcc)
    max_mfcc = np.max(mfcc)
    median_mfcc = np.median(mfcc)
    var_mfcc = np.var(mfcc)
    skewness_mfcc = stats.skew(mfcc.flatten())
    q1_mfcc = np.percentile(mfcc, 25)
    q3_mfcc = np.percentile(mfcc, 75)
    iqr_mfcc = stats.iqr(mfcc.flatten())
    minmax_mfcc = np.ptp(mfcc)
    kurtosis_mfcc = stats.kurtosis(mfcc.flatten())
    
    return {
        'Mean mfcc': mean_mfcc,
        'Std Dev mfcc': std_mfcc,
        'Max mfcc': max_mfcc,
        'Median mfcc': median_mfcc,
        'Variance mfcc': var_mfcc,
        'Skewness mfcc': skewness_mfcc,
        'Quartile 1 mfcc': q1_mfcc,
        'Quartile 3 mfcc': q3_mfcc,
        'IQR mfcc': iqr_mfcc,
        'MinMax mfcc': minmax_mfcc,
        'Kurtosis mfcc': kurtosis_mfcc
    }

# %%
def compute_chroma_stft_features(y, sr):
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    
    mean_chroma_stft = np.mean(chroma_stft)
    std_chroma_stft = np.std(chroma_stft)
    max_chroma_stft = np.max(chroma_stft)
    median_chroma_stft = np.median(chroma_stft)
    var_chroma_stft = np.var(chroma_stft)
    skewness_chroma_stft = stats.skew(chroma_stft.flatten())
    q1_chroma_stft = np.percentile(chroma_stft, 25)
    q3_chroma_stft = np.percentile(chroma_stft, 75)
    iqr_chroma_stft = stats.iqr(chroma_stft.flatten())
    minmax_chroma_stft = np.ptp(chroma_stft)
    kurtosis_chroma_stft = stats.kurtosis(chroma_stft.flatten())
    
    return {
        'Mean chroma_stft': mean_chroma_stft,
        'Std Dev chroma_stft': std_chroma_stft,
        'Max chroma_stft': max_chroma_stft,
        'Median chroma_stft': median_chroma_stft,
        'Variance chroma_stft': var_chroma_stft,
        'Skewness chroma_stft': skewness_chroma_stft,
        'Quartile 1 chroma_stft': q1_chroma_stft,
        'Quartile 3 chroma_stft': q3_chroma_stft,
        'IQR chroma_stft': iqr_chroma_stft,
        'MinMax chroma_stft': minmax_chroma_stft,
        'Kurtosis chroma_stft': kurtosis_chroma_stft
    }

# %%
def compute_chroma_cqt_features(y, sr):
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    mean_chroma_cqt = np.mean(chroma_cqt)
    std_chroma_cqt = np.std(chroma_cqt)
    max_chroma_cqt = np.max(chroma_cqt)
    median_chroma_cqt = np.median(chroma_cqt)
    var_chroma_cqt = np.var(chroma_cqt)
    skewness_chroma_cqt = stats.skew(chroma_cqt.flatten())
    q1_chroma_cqt = np.percentile(chroma_cqt, 25)
    q3_chroma_cqt = np.percentile(chroma_cqt, 75)
    iqr_chroma_cqt = stats.iqr(chroma_cqt.flatten())
    minmax_chroma_cqt = np.ptp(chroma_cqt)
    kurtosis_chroma_cqt = stats.kurtosis(chroma_cqt.flatten())
    
    return {
        'Mean chroma_cqt': mean_chroma_cqt,
        'Std Dev chroma_cqt': std_chroma_cqt,
        'Max chroma_cqt': max_chroma_cqt,
        'Median chroma_cqt': median_chroma_cqt,
        'Variance chroma_cqt': var_chroma_cqt,
        'Skewness chroma_cqt': skewness_chroma_cqt,
        'Quartile 1 chroma_cqt': q1_chroma_cqt,
        'Quartile 3 chroma_cqt': q3_chroma_cqt,
        'IQR chroma_cqt': iqr_chroma_cqt,
        'MinMax chroma_cqt': minmax_chroma_cqt,
        'Kurtosis chroma_cqt': kurtosis_chroma_cqt
    }

# %%
def compute_rms_features(y):
    rms = librosa.feature.rms(y=y)
    
    mean_rms = np.mean(rms)
    std_rms = np.std(rms)
    max_rms = np.max(rms)
    median_rms = np.median(rms)
    var_rms = np.var(rms)
    skewness_rms = stats.skew(rms.flatten())
    q1_rms = np.percentile(rms, 25)
    q3_rms = np.percentile(rms, 75)
    iqr_rms = stats.iqr(rms.flatten())
    minmax_rms = np.ptp(rms)
    kurtosis_rms = stats.kurtosis(rms.flatten())
    
    return {
        'Mean rms': mean_rms,
        'Std Dev rms': std_rms,
        'Max rms': max_rms,
        'Median rms': median_rms,
        'Variance rms': var_rms,
        'Skewness rms': skewness_rms,
        'Quartile 1 rms': q1_rms,
        'Quartile 3 rms': q3_rms,
        'IQR rms': iqr_rms,
        'MinMax rms': minmax_rms,
        'Kurtosis rms': kurtosis_rms
    }

# %%
def define_tempo(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)

    return {
        'Tempo' : tempo[0]
    }

# %% [markdown]
# Mengambil informasi Mel-frequency cepstral coefficients (MFCCs), Chromagram from a waveform or power spectrogram (Chroma STFTs), Constant-Q chromagram (Chroma CQTs), root-mean-square (RMS) value for each frame kemudian mengekstrak nilai-nilai tersebut dengan Mean, Standard Deviation, Maximum, Median, Variance, Skewness, Quartile 1, Quartile 3, IQR, MinMax, dan Kurtosis pada masing-masing informasi, terakhir ditambahkan fitur tempo pada masing-masing audio file

# %%
data = []

for index, row in new_data_labels.iterrows():
    filename = row['filename']
    label = row['label']
    
    # Load the audio file
    y, sr = librosa.load(f"DHD/final_audio/{filename}", sr=None)
    
    # Compute features
    mfcc_features = compute_mfcc_features(y, sr)
    chroma_stft_features = compute_chroma_stft_features(y, sr)
    chroma_cqt_features = compute_chroma_cqt_features(y, sr)
    rms_features = compute_rms_features(y)
    tempo = define_tempo(y, sr)
    
    # Combine features into a single row
    feature_row = {
        'Label': label,
    }
    
    feature_row.update(mfcc_features)
    feature_row.update(chroma_stft_features)
    feature_row.update(chroma_cqt_features)
    feature_row.update(rms_features)
    feature_row.update(tempo)
    data.append(feature_row)

# Create DataFrame from the collected data
df_features = pd.DataFrame(data)

print(df_features.head())

# %%
df_features.to_csv('DHD/df_features.csv', index=False)

# %%
df_features.head()

# %% [markdown]
# Terdapat 45 Fitur

# %% [markdown]
# # Data Preprocessing #2

# %%
df_features = pd.read_csv('DHD/df_features.csv')
df_features.head()

# %%
df_features.info()

# %%
missing_values_count = df_features.isnull().sum()

print(missing_values_count)

# %%
rows_with_nan = df_features[df_features.isnull().any(axis=1)]

rows_with_nan

# %% [markdown]
# Terdapat hanya 1 data yang memiliki nilai NaN saya lebih memilih menghapusnya karena hanya 1 data saja yang memiliki NaN

# %%
# Menghapus baris yang memiliki nilai NaN
df_features = df_features.dropna()

# Menampilkan DataFrame setelah menghapus baris dengan nilai NaN
df_features

# %%
X = df_features.drop(columns='Label')
y = df_features['Label']

# %% [markdown]
# Membagi data label (y) dan data fitur (X)

# %%
le = LabelEncoder()
y = le.fit_transform(y)

# %% [markdown]
# Encode data Label dengan Label Encoder karena mesin hanya bisa membaca data angka

# %%
label_counts = df_features['Label'].value_counts()
label_percentages = label_counts / label_counts.sum() * 100

# %%
# Membuat bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
plt.xlabel('Label')
plt.ylabel('Jumlah')
plt.title('Bar Plot dari Label')
plt.show()

# Membuat pie chart
plt.figure(figsize=(8, 8))
plt.pie(label_percentages, labels=label_percentages.index, autopct='%1.2f%%', colors=sns.color_palette('viridis', len(label_percentages)))
plt.title('Pie Chart dari Label')
plt.show()

# %% [markdown]
# Seperti dijelaskan sebelumnya bahwa data ini tidak seimbang antara data lainnya, oleh karena itu kita wajib melakukan sampling pada kesempatan ini saya menggunakan SMOTE untuk oversampling

# %%
sm = SMOTE(random_state=42)
X_resampling, y_resampling = sm.fit_resample(X, y)

# %%
y_original = le.inverse_transform(y_resampling)

# Menghitung distribusi label asli dalam y_resampling
label_counts_original = Counter(y_original)
label_counts_original = dict(sorted(label_counts_original.items()))

# Membuat bar plot untuk y_resampling dengan label asli
plt.figure(figsize=(10, 6))
sns.barplot(x=list(label_counts_original.keys()), y=list(label_counts_original.values()), palette='viridis')
plt.xlabel('Label Asli')
plt.ylabel('Jumlah')
plt.title('Bar Plot dari Distribusi Label Asli Setelah Resampling')
plt.xticks(rotation=45)  # Memutar label jika panjang
plt.show()

# Menghitung persentase masing-masing kelas dalam y_resampling dengan label asli
label_percentages_original = {label: count / len(y_original) * 100 for label, count in label_counts_original.items()}

# Membuat pie chart untuk y_resampling dengan label asli
plt.figure(figsize=(8, 8))
plt.pie(list(label_percentages_original.values()), labels=list(label_percentages_original.keys()), autopct='%1.2f%%', colors=sns.color_palette('viridis', len(label_percentages_original)))
plt.title('Pie Chart dari Distribusi Label Asli Setelah Resampling')
plt.show()

# %% [markdown]
# Dengan menggunakan SMOTE, data berhasil seimbang antara data lainnya

# %%
X_resampling

# %% [markdown]
# Dan sekarang kita memiliki 1775 data

# %%
pca = PCA(n_components=30)
pca.fit(X_resampling)

# Ambil nilai bobot dari komponen utama
components = pca.components_

# Buat DataFrame untuk menyimpan bobot fitur
weights_df = pd.DataFrame(components, columns=X_resampling.columns)

# Urutkan bobot fitur secara absolut untuk setiap komponen utama
sorted_weights = weights_df.abs().apply(lambda x: x.sort_values(ascending=False), axis=1)

# Ambil 30 fitur dengan bobot tertinggi untuk setiap komponen utama
selected_features = sorted_weights.columns[:30]

print("Fitur yang dipilih oleh PCA:")
print(selected_features)
print(len(selected_features))

# %% [markdown]
# Menggunakan feature selection dengan PCA sebanyak 30 fitur saja yang akan digunakan agar sumber daya yang digunakan lebih efisien dan menghindari overfitting

# %%
X_resampling_pca = pca.transform(X_resampling)
X_resampling_pca = pd.DataFrame(data=X_resampling_pca, index=X_resampling.index)
X_resampling_pca

# %%
X_train, X_test, y_train, y_test = train_test_split(X_resampling_pca, y_resampling, test_size = 0.2, random_state = 42)

# %% [markdown]
# Split data menjadi train dan test

# %%
#proses standarisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
print(type(X_train))
X_train = pd.DataFrame(X_train, columns=X_resampling_pca.columns)

# %%
X_train.head()

# %%
X_train.describe().round(4)

# %% [markdown]
# Standarisasi fitur pada data training

# %%
X_test = scaler.transform(X_test)
print(type(X_test))
X_test = pd.DataFrame(X_test, columns=X_resampling_pca.columns)

# %%
X_test.head()

# %%
X_test.describe().round(4)

# %% [markdown]
# Standarisasi fitur pada data test dari standarisasi yang sudah fit pada data training

# %% [markdown]
# # Model Development

# %% [markdown]
# ## Random Forest

# %% [markdown]
# Menggunakan Algoritma Random Forest

# %%
def grid_search_rf(X, y, param_grid=None, n_jobs=-1):
    # Parameter grid untuk GridSearchCV
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [16, 32, 64, 128, 256],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    # Inisialisasi Random Forest
    rf = RandomForestClassifier(random_state=42)
    
    # Inisialisasi GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=n_jobs)
    
    # Fit GridSearchCV pada seluruh data
    grid_search.fit(X, y)
    
    # Menampilkan parameter terbaik dari GridSearchCV
    print(f'Best Parameters: {grid_search.best_params_}')
    
    # Menampilkan semua parameter yang dicoba
    print('Grid Search Parameters:')
    for i in range(len(grid_search.cv_results_['params'])):
        print(f"Params: {grid_search.cv_results_['params'][i]}, Mean Test Score: {grid_search.cv_results_['mean_test_score'][i] * 100:.2f}%")

# %%
grid_search_rf(X_train, y_train)

# %% [markdown]
# Dengan menggunakan grid search untuk parameter tuning, didapatkan 'max_depth': 32, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200 dengan skor 90.92% Akurasi

# %%
def k_fold_cross_validation_rf(X, y, k=7, n_estimators=200, max_depth=32, min_samples_leaf=1, min_samples_split=2):
    # Inisialisasi KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Daftar untuk menyimpan hasil
    accuracies = []
    sensitivities = []
    precisions = []
    f1_scores = []
    confusion_matrices = []
    models = []
    
    # Loop untuk setiap fold
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f'Fold {fold}')
        
        # Split data menjadi data latih dan data tes
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Inisialisasi dan latih Random Forest
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=42)
        rf.fit(X_train, y_train)
        
        # Prediksi
        y_pred = rf.predict(X_test)
        
        # Hitung metrik
        acc = accuracy_score(y_test, y_pred)
        sensitivity = recall_score(y_test, y_pred, average='macro')  # Sensitivitas untuk multiclass
        precision = precision_score(y_test, y_pred, average='macro')  # Precision untuk multiclass
        f1 = f1_score(y_test, y_pred, average='macro')  # F1 Score untuk multiclass
        
        # Simpan hasil
        accuracies.append(acc * 100)  # Convert to percentage
        sensitivities.append(sensitivity * 100)
        precisions.append(precision * 100)
        f1_scores.append(f1 * 100)
        confusion_matrices.append(confusion_matrix(y_test, y_pred))
        models.append(rf)

        # Tampilkan confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix for Fold {fold}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    # Menampilkan hasil
    print("Akurasi setiap fold (%):", ["{:.2f}".format(acc) for acc in accuracies])
    print("Sensitivitas setiap fold (%):", ["{:.2f}".format(sens) for sens in sensitivities])
    print("Spesifisitas setiap fold (%):", ["{:.2f}".format(spec) for spec in precisions])
    print("F1 Score setiap fold (%):", ["{:.2f}".format(f1) for f1 in f1_scores])

    return models


# %%
models_rf = k_fold_cross_validation_rf(X_train, y_train)

# %% [markdown]
# Menggunakan K-Fold Cross Validation sebanyak 7 dengan waktu komputasi 11.7s dan didapatkan yang terbaik yaitu Fold ke-3 dengan skor Akurasi sebesar 96.06%, Sensitivitas 96%, Spesifisitas 95.89%, dan F1-Score 95.93% pada data training

# %% [markdown]
# Berikut adalah rata-rata untuk masing-masing metrik:
# 
# - **Rata-rata Akurasi:** 90.91%
# - **Rata-rata Sensitivitas:** 90.68%
# - **Rata-rata Spesifisitas:** 90.61%
# - **Rata-rata F1 Score:** 90.21%

# %%
for i in range(len(models_rf)):
    print(f'====Fold {i+1}====')
    # Prediksi
    y_pred = models_rf[i].predict(X_test)

    # Hitung metrik
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Sensitivity: ', recall_score(y_test, y_pred, average='macro'))  # Sensitivitas untuk multiclass
    print('Precision: ', precision_score(y_test, y_pred, average='macro'))  # Precision untuk multiclass
    print('F1 Score: ', f1_score(y_test, y_pred, average='macro'))  # F1 Score untuk multiclass

    # Tampilkan confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix for Fold Random Forest')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# %% [markdown]
# Selanjutnya kita akan memeriksa semua Fold terhadap data yang belum dilihatnya, Fold pertama meraih nilai performa paling tinggi yaitu:
# 
# Accuracy:  91.55%
# Sensitivity:  92.67%
# Precision:  91.58%
# F1 Score:  91.86%

# %% [markdown]
# Berikut adalah rata-rata dari setiap metrik dalam fold:
# <br>
# - Accuracy: 91.05%
# - Sensitivity: 92.01%
# - Precision: 92.10%
# - F1 Score: 92.00%

# %%
#Fungsi untuk plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

# %%
def create_lstm_model(input_shape, num_classes):
    model = models.Sequential()
    
    # LSTM Layer
    model.add(layers.LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(layers.LSTM(64))
    
    # Fully Connected Layer
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# %% [markdown]
# Membuat model Deep Learning dengan arsitekur LSTM

# %%
model = KerasClassifier(build_fn=create_lstm_model, verbose=0, input_shape=(40, 1), num_classes=5)

param_grid = {
    'batch_size': [64, 128, 256],
    'epochs': [50, 100, 150, 200]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)

grid_result = grid.fit(X_train, y_train)

means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']

epochs = [param['epochs'] for param in params]

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for mean, param in zip(means, params):
    print("%f with: %r" % (mean, param))

# %% [markdown]
# Melakukan parameter tuning dengan Grid Search, didapatkan {'batch_size': 64, 'epochs': 200} dengan skor 81.9% Akurasi

# %%
# Fungsi untuk melakukan K-fold cross-validation
def k_fold_cross_validation_lstm(X, y, k=7, epochs=200, batch_size=64):
    # Inisialisasi KFold
    skf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Daftar untuk menyimpan model pada setiap fold dan akurasi, sensitivitas, spesifisitas, F1 score
    models = []
    accuracies = []
    sensitivities = []
    precisions = []
    f1_scores = []
    
    # Loop melalui setiap fold
    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        print(f'Fold {fold}')
        
        # Split data menjadi data latih dan data validasi
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Inisialisasi model
        model = create_lstm_model(input_shape=(40, 1), num_classes=5)
        
        # Latih model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=2)
        
        plot_training_history(history)
        
        # Evaluasi model pada data validasi
        y_pred = model.predict(X_val)
        y_pred_classes = y_pred.argmax(axis=-1)
        
        # Hitung metrik
        accuracy = accuracy_score(y_val, y_pred_classes)
        cm = confusion_matrix(y_val, y_pred_classes)
        sensitivity = recall_score(y_val, y_pred_classes, average='macro')
        precision = precision_score(y_val, y_pred_classes, average='macro')
        f1 = f1_score(y_val, y_pred_classes, average='macro')
        
        # Tambahkan metrik ke dalam daftar
        accuracies.append(accuracy * 100)
        sensitivities.append(sensitivity * 100)
        precisions.append(precision * 100)
        f1_scores.append(f1 * 100)
        
        # Tambahkan model ke dalam daftar
        models.append(model)
        
        print(f'Validation Accuracy: {accuracy * 100:.2f}%')
        print()
    
    # Menampilkan hasil
    print("Akurasi setiap fold (%):", ["{:.2f}".format(acc) for acc in accuracies])
    print("Sensitivitas setiap fold (%):", ["{:.2f}".format(sens) for sens in sensitivities])
    print("Spesifisitas setiap fold (%):", ["{:.2f}".format(spec) for spec in precisions])
    print("F1 Score setiap fold (%):", ["{:.2f}".format(f1) for f1 in f1_scores])
    
    return models


# %%
models_lstm = k_fold_cross_validation_lstm(X_train, y_train)

# %% [markdown]
# Dilakukan K-Fold Cross Validation sebanyak 7 dengan waktu komputasi 20m 30.9s, didapatkan Fold ketiga dengan nilai terbaik yaitu Akurasi 85.71%, Sensitivitas 84.94%, Spesifitas 85.06%, dan F1 Score 84.87%

# %% [markdown]
# Berikut adalah rata-rata untuk masing-masing metrik:
# <br>
# - **Akurasi:** 79.64%
# - **Sensitivitas:** 79.31%
# - **Spesifisitas:** 78.89%
# - **F1 Score:** 78.39%

# %%
for i in range(len(models_lstm)):
    print(f'====Fold {i+1}====')
    # Prediksi
    y_pred_prob = models_lstm[i].predict(X_test)
    y_pred = y_pred_prob.argmax(axis=-1)

    # Hitung metrik
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Sensitivity: ', recall_score(y_test, y_pred, average='macro'))  # Sensitivitas untuk multiclass
    print('Precision: ', precision_score(y_test, y_pred, average='macro'))  # Precision untuk multiclass
    print('F1 Score: ', f1_score(y_test, y_pred, average='macro'))  # F1 Score untuk multiclass

    # Tampilkan confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix for Fold Random Forest')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# %% [markdown]
# Dengan mengujikan semua model Fold pada data yang belum dilihat, model Fold ke-7 memiliki performa terbaik yaitu:
# 
# Accuracy:  82.82%
# Sensitivity:  84.33%
# Precision:  83.05%
# F1 Score:  83.38%

# %% [markdown]
# Berikut adalah rata-rata untuk masing-masing metrik:
# 
# - **Akurasi:** 76.74%
# - **Sensitivitas:** 78.93%
# - **Presisi:** 77.14%
# - **F1 Score:** 77.07%

# %%
import pickle

#simpan model random forest
with open('DHD/models_rf.pkl', 'wb') as f:
    pickle.dump(models_rf, f)

#simpan model lstm
with open('DHD/models_lstm.pkl', 'wb') as f:
    pickle.dump(models_lstm, f)

# %% [markdown]
# # Kesimpulan

# %% [markdown]
# Model Random Forest mengungguli Model LSTM dengan skor Sensitivity:  92.67% dan Precision:  91.58% dibanding LSTM dengan skor Sensitivity:  84.33% dan Precision:  83.05% pada data yang belum dilihat, sekaligus mengungguli sumber daya komputasi dengan alat ukur waktu pada kasus K-Fold Cross Validation sebanyak 7 yaitu Random Forest menghabiskan waktu 11.7s sementara LSTM 20m 30.9s


