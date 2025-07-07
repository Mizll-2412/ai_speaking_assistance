import pandas as pd
import numpy as np
import librosa
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
import warnings
warnings.filterwarnings('ignore')


#1. Tien xu ly du lieu
class AudioProcessor:
    def __init__(self, sr=16000, n_mfcc=13, max_length=800):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_length = max_length
    def extract_features(self, audio_bytes):
        try:
            audio_data,_ =  librosa.load(io.BytesIO(audio_bytes), sr= self.sr)
            mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sr, n_mfcc=self.n_mfcc)
            mfcc = mfcc.T
            if len(mfcc)> self.max_length:
                mfcc = mfcc[:self.max_length]
            else:
                padding = np.zeros((self.max_length-len(mfcc), self.n_mfcc))
                mfcc = np.vstack([mfcc, padding])
            return mfcc
        except Exception as e:
            print("Error process audio {e}")
            return np.zeros((self.max_length, self.n_mfcc))
def preprocess_data(df):
    print("Starting preprocess data")
    processor = AudioProcessor()
    features = []
    for i, audio_data in enumerate(tqdm(df['audio'], desc="Extract Features")):
        if isinstance(audio_data, dict) and 'bytes' in audio_data:
            feature = processor.extract_features(audio_data['bytes'])
        else:
            feature = np.zeros((processor.max_length, processor.n_mfcc))
        features.append(feature)
    X=np.array(features)
    score_columns = ['accuracy', 'completeness', 'fluency', 'prosodic']
    y = df[score_columns].values.astype(np.float32)
    print(f"X shape: {X.shape}")
    print(f"Y shape: {y.shape}")

    return X, y, score_columns

class PreprocessingData:
    def __init__(self):
        self.label_enconder = LabelEncoder()
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
                'completeness': item['completeness'],
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
    def extract_word_features(self,data):
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
                    'min_phone_accuracy':np.min(word['phones-accuracy']),
                    'max_phone_accuracy':np.max(word['phones-accuracy']),
                    'has_mispronunciation': len(word['mispronunciations']) > 0,
                    'phones':word['phones']
                }
                word_features.append(features)
        return pd.DataFrame(word_features)
    def extract_phoneme_features(self, data):
        phoneme_features=[]
        for item in data:
            speaker_id = item['speaker']
            for word in item['words']:
                word_text = word['text']
                for i, (phone, accuracy) in enumerate(zip(word['phones'], word['phones-accuracy'])):
                    features = {
                        'speaker_id': speaker_id,
                        'word_text': word_text,
                        'phone':phone,
                        'phone_accuracy': accuracy,
                        'phone_position':i, 
                        'is_first_phone':i==0,
                        'is_last_phone': i==len(word['phones'])-1,
                        'phone_type': self.classify_phone_type(phone)
                    }
                    phoneme_features.append(features)
        return pd.DataFrame(phoneme_features)
    def classify_phone_type(self, phone):
        vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
        consonants = ['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']

        phone_base = phone.replace('0','').replace('1','').replace('2','')
        if phone_base in vowels:
            return 'vowel'
        elif phone_base in consonants:
            return 'consonant'
        else:
            return 'unknown'
    def extrac_audio_features(self, audio_data, sr=16000):
        features = []
        features["duration"] = len(audio_data)/sr
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
    def normalize_scores(self, df, score_columns):
        df_normalize = df.copy()

        for col in df_normalize:
            if col in df.columns:
                df_normalize[f'{col}_normalized'] = df[col]/10.0

        return df_normalize
    def encode_categorical_features(self, df):
        df_encoded = df.copy()
        categorical_columns = ['gender', 'phone_type']

        for col in categorical_columns:
            if col in df.columns:
                df_encoded[f'{col}_encoded'] = self.label_enconder.fit_transform(df[col])
        return df_encoded
    
    def create_quality_labels(self, df):
        df_labeled = df.copy()
        
        def classify_quality(score):
            if score >= 8:
                return 'excellent'
            elif score >= 6:
                return 'good'
            elif score >=4:
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

                lower_bound = Q1 - 1.5*IQR
                upper_bound = Q3 + 1.5*IQR
                outliers[col] = df[(df[col]<lower_bound)|(df[col] > upper_bound)].index.tolist()
        return outliers
    
    def create_age_group(self, df):
        df_grouped = df.copy()
        def classify_age(age):
            if age<=8:
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
    
#2 Dataset and dataloader
class AudioScoringDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#3 Neural network model
class PronunciationScoringModel(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, num_scores=4, max_length=800):
        super(PronunciationScoringModel, self).__init__()
        
        # CNN layers
        self.conv1=nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2=nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool=nn.MaxPool1d(2)
        #LSTM layers
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first= True,
            dropout=0.3,
            bidirectional=True
        )
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim = hidden_size*2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.fc1=nn.Linear(hidden_size*2, 256)
        self.fc2=nn.Linear(256, 128)
        self.fc3=nn.Linear(128, num_scores)
        self.dropout= nn.Dropout(0.3)
        self.relu=nn.ReLU()
        self.batch_norm1=nn.BatchNorm1d(256)
        self.batch_norm2=nn.BatchNorm1d(128)

    def forward(self, x):
        #Cnn process
        x=x.transpose(1,2)
        x=self.relu(self.conv1(x))
        x=self.pool(x)
        x=self.relu(self.conv2(x))
        x=self.pool(x)
        x=x.transpose(1,2)
        
        #lstm processing
        lstm_out, _= self.lstm(x)

        #attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # golden avg pooling
        pooled = torch.mean(attn_out, dim=1)
        # fully connected layers
        out = self.relu(self.batch_norm1(self.fc1(pooled)))
        out = self.dropout(out)
        out = self.relu(self.batch_norm2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)
        return out
#4 Training model
def training_model(model, train_loader, val_loader, num_epochs=50,learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses=[]
    val_losses=[]
    best_val_loss=float('inf')

    print(f"Training in {device}")
    for epoch in range(num_epochs):
        #training
        model.train()
        train_loss=0.0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs=model(batch_x)
            loss=criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
        #valid
        model.eval()
        val_loss=0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs=model(batch_x)
                val_loss+=criterion(outputs, batch_y).item()
        # caculate avg losses
        train_loss=train_loss/len(train_loader)
        val_loss= val_loss/len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        scheduler.step(val_loss)
        #save best model
        if val_loss < best_val_loss:
            best_val_loss= val_loss
            torch.save(model.state_dict(), 'best_pronunciation_model.pth')
    return train_losses, val_losses
def evaluate_model(model, test_loader, score_columns):
    device = torch.device('cuba' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs= model(batch_x)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    predictions=np.vstack(all_predictions)
    targets=np.vstack(all_targets)
    
    mse= mean_squared_error(targets, predictions)
    mae= mean_absolute_error(targets, predictions)
    r2 = r2_score(targets,predictions)

    print(f"\nResult:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")

    #evaluate for each score
    for i, col in enumerate(score_columns):
        col_mse=mean_squared_error(targets[:, i], predictions[:, i])
        col_mae=mean_absolute_error(targets[:, i], predictions[:, i])
        col_r2=r2_score(targets[:,i], predictions[:,i])
        print(f"{col} - MSE: {col_mse:.4f}, MAE: {col_mae:.4f}, R2: {col_r2:.4f}")
    
    return predictions, targets
def plot_resultt(train_losses, val_losses, predictions, targets, score_columns):
    fig, ax = plt.subplots(2, 3, figsize=(15,10))
    ax[0,0].plot(train_losses, label='Train loss')
    ax[0,0].plot(val_losses, label='Validation loss')
    ax[0,0].set_title('Training history')
    ax[0,0].set_xlabel('Epoch')  
    ax[0,0].set_ylabel('Loss')
    ax[0,0].legend()

    for i, col in enumerate(score_columns):
        row = (i+1)//3
        col_idx = (i+1)%3
        ax[row, col_idx].scatter(targets[:,i], predictions[:,i], alpha=0.6)
        ax[row, col_idx].plot([targets[:, i].min(), targets[:, i].max()], 
                              [targets[:, i].min(), targets[:, i].max()], 'r--')
        ax[row, col_idx].set_title(f'{col}: Predicted vs Actual')
        ax[row, col_idx].set_xlabel('Actual')
        ax[row, col_idx].set_ylabel('Predicted')

    ax[row, col_idx].plot([targets[:, i].min(), targets[:, i].max()], 
                              [targets[:, i].min(), targets[:, i].max()], 'r--')
    ax[row, col_idx].set_title(f'{col}: Predicted vs Actual')
    ax[row, col_idx].set_xlabel('Actual')
    ax[row, col_idx].set_ylabel('Predicted')

def main(df):
    X, y, score_columns = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    train_dataset = AudioScoringDataset(X_train, y_train)
    val_dataset = AudioScoringDataset(X_val, y_val)
    test_dataset = AudioScoringDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = PronunciationScoringModel(
        input_size=13,
        hidden_size=128,
        num_layers=2,
        num_scores=4,
        max_length=800
    )
    print("Model structure")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    # training
    train_losses, val_losses = training_model(
        model, train_loader, val_loader,
        num_epochs=50, learning_rate=0.01
    )
    model.load_state_dict(torch.load('best_pronunciation_model.pth'))
    predictions, targets = evaluate_model(model, test_loader, score_columns)

    plot_resultt(train_losses, val_losses, predictions, targets, score_columns)
    return model, predictions, targets



def predict_pronunciation_score(model, audio_bytes):
    processer = AudioProcessor()
    features = processer.extract_features(audio_bytes)
    features = torch.FloatTensor(features).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        scores = model(features)
    return scores.cpu().numpy()[9]
if __name__ == "__main__":
    df = pd.read_csv('speechocean762.csv')
    X,y, score_columns = preprocess_data(df)
    print(X[5])
    print(y[5])
    # model, predictions, targets = main(df)