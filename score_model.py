import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
from preprocess_score_model import AudioProcessor
warnings.filterwarnings('ignore')

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
    def prediction(self, audio_file_path):
        
        try:
            processor = AudioProcessor()
            features = processor.extract_features_from_file(audio_file_path)
            
            audio_data, sr = librosa.load(audio_file_path, sr=16000)
            duration = len(audio_data) / sr
            estimated_word_count = max(1, int(duration * 2))             
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            word_count_tensor = torch.FloatTensor([estimated_word_count])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.to(device)
            features_tensor = features_tensor.to(device)
            word_count_tensor = word_count_tensor.to(device)
            self.eval()
            with torch.no_grad():
                if self.use_word_count:
                    scores = self(features_tensor, word_count_tensor)
                else:
                    scores = self(features_tensor)
            accuracy = float(scores[0][0].cpu().numpy())            
            accuracy = max(0, min(100, accuracy * 10))
            return estimated_word_count, accuracy
        except Exception as e:
            print(f"Lỗi trong prediction: {e}")
            return 1, 50.0

