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

# 1. Tien xu ly du lieu
class AudioProcessor:
    def __init__(self, sr=16000, n_mfcc=13, max_length=800):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_length = max_length

    def extract_features(self, audio_bytes):
        try:
            audio_data, _ = librosa.load(io.BytesIO(audio_bytes), sr=self.sr)

            # Basic audio augmentation
            if np.random.random() < 0.3:  # 30% chance of augmentation
                # Add slight noise
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
    required_columns = ['audio', 'accuracy', 'fluency', 'prosodic']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    processor = AudioProcessor()
    features = []
    word_counts = []
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

            # Extract word count from the 'words' column
            if 'words' in row and row['words'] is not None:
                word_count = len(row['words'])
            elif 'text' in row and row['text'] is not None:
                word_count = len(row['text'].split())
            else:
                word_count = 0

            word_counts.append(word_count)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            feature = np.zeros((processor.max_length, processor.n_mfcc))
            word_counts.append(0)


        features.append(feature)

    print(f"Successfully processed {valid_samples}/{len(df)} audio samples")

    X = np.array(features)
    score_columns = ['accuracy', 'fluency', 'prosodic']
    y = df[score_columns].values.astype(np.float32)
    word_counts = np.array(word_counts) # Convert word_counts to numpy array

    # Validate shapes
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Feature and target shape mismatch: {X.shape[0]} vs {y.shape[0]}")

    print(f"X shape: {X.shape}")
    print(f"Y shape: {y.shape}")
    print(f"Word counts shape: {word_counts.shape}")
    print(f"Word count statistics: min={word_counts.min()}, max={word_counts.max()}, mean={word_counts.mean():.2f}")
    print(f"Score statistics:")
    for i, col in enumerate(score_columns):
        print(f"  {col}: min={y[:, i].min():.2f}, max={y[:, i].max():.2f}, mean={y[:, i].mean():.2f}")


    return X, y, score_columns,word_counts

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


# 2 Dataset and dataloader
class AudioScoringDataset(Dataset):
    def __init__(self, X, y, word_counts=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.word_counts = torch.FloatTensor(word_counts) if word_counts is not None else None


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.word_counts is not None:
            return self.X[idx], self.y[idx], self.word_counts[idx]
        else:
            return self.X[idx], self.y[idx]

# 3 Neural network model
class PronunciationScoringModel(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, num_scores=4, max_length=800, use_word_count=True):
        super(PronunciationScoringModel, self).__init__()
        self.use_word_count = use_word_count
        # CNN layers
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        fc_input_size = hidden_size * 2
        if use_word_count:
            fc_input_size += 1

        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_scores)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)

    def forward(self, x, word_count = None):
        # CNN process
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.transpose(1, 2)

        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global avg pooling
        pooled = torch.mean(attn_out, dim=1)

        if self.use_word_count and word_count is not None:
            word_count = word_count.unsqueeze(1) if word_count.dim() == 1 else word_count
            pooled = torch.cat([pooled, word_count], dim=1)

        # Fully connected layers
        out = self.relu(self.batch_norm1(self.fc1(pooled)))
        out = self.dropout(out)
        out = self.relu(self.batch_norm2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)
        return out


# 4 Training model
def training_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, patience=10, use_word_count = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Training on {device}")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            if use_word_count and len(batch_data) == 3:
                batch_x, batch_y, batch_word_count = batch_data
                batch_x, batch_y, batch_word_count = batch_x.to(device), batch_y.to(device), batch_word_count.to(device)
            else:
                batch_x, batch_y = batch_data
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_word_count = None

            optimizer.zero_grad()
            outputs = model(batch_x, batch_word_count)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                if use_word_count and len(batch_data) == 3:
                    batch_x, batch_y, batch_word_count = batch_data
                    batch_x, batch_y, batch_word_count = batch_x.to(device), batch_y.to(device), batch_word_count.to(device)
                else:
                    batch_x, batch_y = batch_data
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    batch_word_count = None

                outputs = model(batch_x, batch_word_count)
                val_loss += criterion(outputs, batch_y).item()

        # Calculate avg losses
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        # Save best model and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_pronunciation_model_with_word_numbers.pth')
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered! No improvement for {patience} epochs.")
            break

    return train_losses, val_losses

def evaluate_model(model, test_loader, score_columns, use_word_count=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []
    all_word_counts = []

    with torch.no_grad():
        for batch_data in test_loader:
            if use_word_count and len(batch_data) == 3:
                batch_x, batch_y, batch_word_count = batch_data
                batch_x, batch_y, batch_word_count = batch_x.to(device), batch_y.to(device), batch_word_count.to(device)
                all_word_counts.append(batch_word_count.cpu().numpy())
            else:
                batch_x, batch_y = batch_data
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_word_count = None

            outputs = model(batch_x, batch_word_count)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

    if all_word_counts:
        word_counts = np.concatenate(all_word_counts)
    else:
        word_counts = None


    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    print(f"\nResult:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")

    # Evaluate for each score
    for i, col in enumerate(score_columns):
        col_mse = mean_squared_error(targets[:, i], predictions[:, i])
        col_mae = mean_absolute_error(targets[:, i], predictions[:, i])
        col_r2 = r2_score(targets[:, i], predictions[:, i])
        print(f"{col} - MSE: {col_mse:.4f}, MAE: {col_mae:.4f}, R2: {col_r2:.4f}")

    return predictions, targets, word_counts



def plot_result(train_losses, val_losses, predictions, targets, score_columns):
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # Plot training history
    ax[0, 0].plot(train_losses, label='Train loss')
    ax[0, 0].plot(val_losses, label='Validation loss')
    ax[0, 0].set_title('Training history')
    ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].legend()

    # Plot predictions vs actual for each score
    for i, col in enumerate(score_columns):
        row = (i + 1) // 3
        col_idx = (i + 1) % 3
        ax[row, col_idx].scatter(targets[:, i], predictions[:, i], alpha=0.6)
        ax[row, col_idx].plot([targets[:, i].min(), targets[:, i].max()],
                              [targets[:, i].min(), targets[:, i].max()], 'r--')
        ax[row, col_idx].set_title(f'{col}: Predicted vs Actual')
        ax[row, col_idx].set_xlabel('Actual')
        ax[row, col_idx].set_ylabel('Predicted')
        ax[row, col_idx].legend()

    # Remove empty subplot
    if len(score_columns) < 4:
        ax[1, 2].remove()

    plt.tight_layout()
    plt.show()


def main(df, use_word_count=True):
    X, y, score_columns, word_counts = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Split word counts accordingly
    if use_word_count:
        wc_train, wc_test = train_test_split(word_counts, test_size=0.2, random_state=42)
        wc_train, wc_val = train_test_split(wc_train, test_size=0.2, random_state=42)
    else:
        wc_train = wc_val = wc_test = None

    # Create datasets
    train_dataset = AudioScoringDataset(X_train, y_train, wc_train)
    val_dataset = AudioScoringDataset(X_val, y_val, wc_val)
    test_dataset = AudioScoringDataset(X_test, y_test, wc_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create model
    model = PronunciationScoringModel(
        input_size=13,
        hidden_size=128,
        num_layers=2,
        num_scores=len(score_columns),
        max_length=800,
        use_word_count=use_word_count
    )

    print("Model structure")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Training
    train_losses, val_losses = training_model(
        model, train_loader, val_loader,
        num_epochs=50, learning_rate=0.001, patience=10,
        use_word_count=use_word_count
    )

    # Load best model and evaluate
    model.load_state_dict(torch.load('best_pronunciation_model_with_word_numbers.pth'))
    predictions, targets, test_word_counts = evaluate_model(
        model, test_loader, score_columns, use_word_count=use_word_count
    )

    # Create results dictionary
    results_dict = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'predictions': predictions,
        'targets': targets,
        'test_word_counts': test_word_counts,
        'score_columns': score_columns,
        'word_count_stats': {
            'min': word_counts.min(),
            'max': word_counts.max(),
            'mean': word_counts.mean(),
            'std': word_counts.std()
        }
    }

    # Print word count analysis
    if test_word_counts is not None:
        print(f"\nWord Count Analysis:")
        print(f"Test set word counts - Min: {test_word_counts.min()}, Max: {test_word_counts.max()}, Mean: {test_word_counts.mean():.2f}")

        # Analyze performance by word count ranges
        wc_ranges = [(0, 5), (6, 10), (11, 15), (16, 20), (21, float('inf'))]
        for wc_min, wc_max in wc_ranges:
            if wc_max == float('inf'):
                mask = test_word_counts >= wc_min
                range_name = f"{wc_min}+"
            else:
                mask = (test_word_counts >= wc_min) & (test_word_counts <= wc_max)
                range_name = f"{wc_min}-{wc_max}"

            if np.sum(mask) > 0:
                range_mse = mean_squared_error(targets[mask], predictions[mask])
                range_mae = mean_absolute_error(targets[mask], predictions[mask])
                print(f"Word count {range_name}: {np.sum(mask)} samples, MSE: {range_mse:.4f}, MAE: {range_mae:.4f}")


    # Plot results
    plot_result(train_losses, val_losses, predictions, targets, score_columns)

    return model, predictions, targets, word_counts, results_dict


def predict_pronunciation_score(model, audio_bytes):
    processor = AudioProcessor()
    features = processor.extract_features(audio_bytes)
    features = torch.FloatTensor(features).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    features = features.to(device)

    model.eval()
    with torch.no_grad():
        scores = model(features)
    return scores.cpu().numpy()[0]


if __name__ == "__main__":
    dataset = load_dataset(
    'mispeech/speechocean762',
    cache_dir='/tmp/datasets_cache'
    )
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    if 'completeness' in combined_df.columns:
        combined_df = combined_df.drop('completeness', axis=1)

    model, predictions, targets, word_counts, results = main(combined_df, use_word_count=True)