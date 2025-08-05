import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import io
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

class AudioProcessor:
    def __init__(self, sr=16000, n_mfcc=13, max_length=800):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_length = max_length

    def extract_features(self, audio_bytes):
        try:
            audio_data, _ = librosa.load(io.BytesIO(audio_bytes), sr=self.sr)

            if np.random.random() < 0.3: 
                noise = np.random.normal(0, 0.005, audio_data.shape)
                audio_data = audio_data + noise

            mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sr, n_mfcc=self.n_mfcc)
            mfcc = mfcc.T

            if len(mfcc) > self.max_length:
                mfcc = mfcc[:self.max_length]
            else:
                padding = np.zeros((self.max_length - len(mfcc), self.n_mfcc))
                mfcc = np.vstack([mfcc, padding])

            return mfcc
        except Exception as e:
            print(f"Error process audio {e}")
            return np.zeros((self.max_length, self.n_mfcc))

    def extract_features_from_file(self, audio_path):
        """Extract features directly from audio file path"""
        try:
            audio_data, _ = librosa.load(audio_path, sr=self.sr)

            # Convert to bytes for consistent processing
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, self.sr, format='wav')
            buffer.seek(0)

            return self.extract_features(buffer.read())
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return np.zeros((self.max_length, self.n_mfcc))

def preprocess_data(df):
    print("Starting preprocess data")

    # Validate input data
    required_columns = ['audio', 'accuracy', 'fluency', 'prosodic', 'completeness']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    processor = AudioProcessor()
    features = []
    valid_samples = 0

    for i, row in df.iterrows():
        audio_data = row['audio']
        try:
            if isinstance(audio_data, dict) and 'bytes' in audio_data:
                feature = processor.extract_features(audio_data['bytes'])
                valid_samples += 1
            elif isinstance(audio_data, str):  # If audio_data is a file path
                # Load audio file from path
                try:
                    audio_array, sr = librosa.load(audio_data, sr=processor.sr)
                    # Convert to bytes-like format for processing
                    import soundfile as sf
                    import io
                    buffer = io.BytesIO()
                    sf.write(buffer, audio_array, sr, format='wav')
                    buffer.seek(0)
                    feature = processor.extract_features(buffer.read())
                    valid_samples += 1
                except Exception as e:
                    print(f"Error loading audio file {audio_data}: {e}")
                    feature = np.zeros((processor.max_length, processor.n_mfcc))
            else:
                feature = np.zeros((processor.max_length, processor.n_mfcc))
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            feature = np.zeros((processor.max_length, processor.n_mfcc))


        features.append(feature)

    print(f"Successfully processed {valid_samples}/{len(df)} audio samples")

    X = np.array(features)
    score_columns = ['accuracy', 'fluency', 'prosodic', 'completeness']
    y = df[score_columns].values.astype(np.float32)

    # Validate shapes
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Feature and target shape mismatch: {X.shape[0]} vs {y.shape[0]}")

    print(f"X shape: {X.shape}")
    print(f"Y shape: {y.shape}")
    print(f"Score statistics:")
    for i, col in enumerate(score_columns):
        print(f"  {col}: min={y[:, i].min():.2f}, max={y[:, i].max():.2f}, mean={y[:, i].mean():.2f}")


    return X, y, score_columns

class PreprocessingData:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.audio_features = []


    def extract_sentence_features(self, data):
        sentence_features = []
        for item in data:
            features = {
                'speaker_id': item['speaker'],
                'gender': item['gender'],
                'age': item['age'],
                'text': item['text'],
                'accuracy': item['accuracy'],
                'fluency': item['fluency'],
                'prosodic': item['prosodic'],
                'total_score': item['total'],
                'audio_path': item['audio']['path'],
                'sampling_rate': item['audio']['sampling_rate'],
                'audio_length': len(item['audio']['array']),
                'num_words': len(item['words']),
                'text_length': len(item['text'].split())
            }
            sentence_features.append(features)
        return pd.DataFrame(sentence_features)

    def extract_word_features(self, data):
        word_features = []
        for item in data:
            speaker_id = item['speaker']
            for word in item['words']:
                features = {
                    'speaker_id': speaker_id,
                    'word_text': word['text'],
                    'word_accuracy': word['accuracy'],
                    'word_stress': word['stress'],
                    'word_total': word['total'],
                    'num_phones': len(word['phones']),
                    'avg_phone_accuracy': np.mean(word['phones-accuracy']),
                    'min_phone_accuracy': np.min(word['phones-accuracy']),
                    'max_phone_accuracy': np.max(word['phones-accuracy']),
                    'has_mispronunciation': len(word['mispronunciations']) > 0,
                    'phones': word['phones']
                }
                word_features.append(features)
        return pd.DataFrame(word_features)

    def extract_phoneme_features(self, data):
        phoneme_features = []
        for item in data:
            speaker_id = item['speaker']
            for word in item['words']:
                word_text = word['text']
                for i, (phone, accuracy) in enumerate(zip(word['phones'], word['phones-accuracy'])):
                    features = {
                        'speaker_id': speaker_id,
                        'word_text': word_text,
                        'phone': phone,
                        'phone_accuracy': accuracy,
                        'phone_position': i,
                        'is_first_phone': i == 0,
                        'is_last_phone': i == len(word['phones']) - 1,
                        'phone_type': self.classify_phone_type(phone)
                    }
                    phoneme_features.append(features)
        return pd.DataFrame(phoneme_features)

    def classify_phone_type(self, phone):
        vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
        consonants = ['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']

        phone_base = phone.replace('0', '').replace('1', '').replace('2', '')
        if phone_base in vowels:
            return 'vowel'
        elif phone_base in consonants:
            return 'consonant'
        else:
            return 'unknown'

    def extract_audio_features(self, audio_data, sr=16000):
        features = {}
        features["duration"] = len(audio_data) / sr
        features["rms_energy"] = np.sqrt(np.mean(audio_data**2))
        features["zero_crossing_rate"] = np.mean(librosa.feature.zero_crossing_rate(audio_data))

        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        features["mfcc_mean"] = np.mean(mfccs, axis=1)
        features["mfcc_std"] = np.std(mfccs, axis=1)

        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        features["spectral_centroid_mean"] = np.mean(spectral_centroids)
        features["spectral_centroid_std"] = np.std(spectral_centroids)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)

        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        features["chroma_mean"] = np.mean(chroma, axis=1)
        features["chroma_std"] = np.std(chroma, axis=1)

        return features

    def preprocess_for_model(self, raw_data):
        X = []
        y = []
        for item in raw_data:
            audio_features = self.extract_audio_features(item['audio']['array'], item['audio']['sampling_rate'])
            feature_vector = []
            feature_vector.extend(audio_features["mfcc_mean"])
            feature_vector.extend(audio_features["mfcc_std"])
            feature_vector.extend([
                audio_features['duration'],
                audio_features['rms_energy'],
                audio_features['zero_crossing_rate'],
                audio_features['spectral_centroid_mean'],
                audio_features['spectral_centroid_std'],
                audio_features['spectral_rolloff_mean'],
                audio_features['spectral_rolloff_std']
            ])
            feature_vector.extend(audio_features['chroma_mean'])
            feature_vector.extend(audio_features['chroma_std'])
            feature_vector.extend([
                item['age'],
                1 if item['gender'] == 'm' else 0,
                len(item['words']),
                len(item['text'].split())
            ])
            X.append(feature_vector)
            scores = [
                item['accuracy'] / 10.0,
                item['fluency'] / 10.0,
                item['prosodic'] / 10.0
            ]
            y.append(scores)
        X = np.array(X)
        y = np.array(y)
        return X, y

    def normalize_scores(self, df, score_columns):
        df_normalize = df.copy()
        for col in score_columns:
            if col in df.columns:
                df_normalize[f'{col}_normalized'] = df[col] / 10.0
        return df_normalize

    def encode_categorical_features(self, df):
        df_encoded = df.copy()
        categorical_columns = ['gender', 'phone_type']

        for col in categorical_columns:
            if col in df.columns:
                df_encoded[f'{col}_encoded'] = self.label_encoder.fit_transform(df[col])
        return df_encoded

    def create_quality_labels(self, df):
        df_labeled = df.copy()

        def classify_quality(score):
            if score >= 8:
                return 'excellent'
            elif score >= 6:
                return 'good'
            elif score >= 4:
                return 'fair'
            else:
                return 'poor'

        if 'total_score' in df.columns:
            df_labeled['quality_label'] = df['total_score'].apply(classify_quality)
        return df_labeled

    def detect_outliers(self, df, columns):
        outliers = {}
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        return outliers

    def create_age_group(self, df):
        df_grouped = df.copy()

        def classify_age(age):
            if age <= 8:
                return 'young_child'
            elif age <= 12:
                return 'child'
            elif age <= 17:
                return 'teenager'
            else:
                return 'adult'

        if 'age' in df.columns:
            df_grouped['age_group'] = df['age'].apply(classify_age)
        return df_grouped

    def split_data(self, df, stratify_column=None, test_size=0.2, val_size=0.2):
        """Chia dữ liệu thành train/val/test"""
        if stratify_column and stratify_column in df.columns:
            stratify = df[stratify_column]
        else:
            stratify = None

        # Chia train và temp (test + val)
        train_df, temp_df = train_test_split(
            df, test_size=test_size + val_size,
            random_state=42, stratify=stratify
        )

        # Chia temp thành val và test
        val_df, test_df = train_test_split(
            temp_df, test_size=test_size / (test_size + val_size),
            random_state=42
        )

        return train_df, val_df, test_df

    def preprocess_pipeline(self, raw_data):
        # 1 trich xuat dac trung
        print("1. Trich xuat dac trung")
        sentence_df = self.extract_sentence_features(raw_data)
        word_df = self.extract_word_features(raw_data)
        phoneme_df = self.extract_phoneme_features(raw_data)

        # 2 chuan hoa diem so
        print("2. Chuan hoa diem so")
        score_columns = ['accuracy', 'fluency', 'prosodic', 'total_score']
        sentence_df = self.normalize_scores(sentence_df, score_columns)
        word_score_columns = ['word_accuracy', 'word_stress', 'word_total']
        word_df = self.normalize_scores(word_df, word_score_columns)

        # 3 ma hoa dac trung phan loai
        print("3. Ma hoa dac trung phan loai")
        sentence_df = self.encode_categorical_features(sentence_df)
        phoneme_df = self.encode_categorical_features(phoneme_df)

        # 4 tao nhan chat luong
        print("4. Tao nhan chat luong")
        sentence_df = self.create_quality_labels(sentence_df)

        # 5 tao nhom tuoi
        print("5 tao nhom tuoi")
        sentence_df = self.create_age_group(sentence_df)

        print("6. Phát hiện ngoại lệ...")
        numeric_columns = ['accuracy', 'fluency', 'prosodic', 'total_score', 'age']
        outliers = self.detect_outliers(sentence_df, numeric_columns)

        # 7. Chia dữ liệu
        print("7. Chia dữ liệu...")
        train_df, val_df, test_df = self.split_data(sentence_df, stratify_column='quality_label')

        print("Hoàn thành tiền xử lý!")

        return {
            'sentence_features': sentence_df,
            'word_features': word_df,
            'phoneme_features': phoneme_df,
            'train_data': train_df,
            'val_data': val_df,
            'test_data': test_df,
            'outliers': outliers
        }
