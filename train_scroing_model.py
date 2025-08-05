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
from preprocess_score_model import AudioProcessor, preprocess_data
from score_model import AudioScoringDataset, PronunciationScoringModel

def training_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, patience=10):
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
            batch_x, batch_y = batch_data
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
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
                batch_x, batch_y = batch_data
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
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
            torch.save(model.state_dict(), 'best_pronunciation_model.pth')
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered! No improvement for {patience} epochs.")
            break

    return train_losses, val_losses

def evaluate_model(model, test_loader, score_columns):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_data in test_loader:
            
            batch_x, batch_y = batch_data
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

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

    return predictions, targets



def plot_result(train_losses, val_losses, predictions, targets, score_columns):
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    ax[0, 0].plot(train_losses, label='Train loss')
    ax[0, 0].plot(val_losses, label='Validation loss')
    ax[0, 0].set_title('Training history')
    ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].legend()

    for i, col in enumerate(score_columns):
        row, col_idx = divmod(i + 1, 3)
        ax[row, col_idx].scatter(targets[:, i], predictions[:, i], alpha=0.6)
        ax[row, col_idx].plot(
            [targets[:, i].min(), targets[:, i].max()],
            [targets[:, i].min(), targets[:, i].max()],
            'r--'
        )
        ax[row, col_idx].set_title(f'{col}: Predicted vs Actual')
        ax[row, col_idx].set_xlabel('Actual')
        ax[row, col_idx].set_ylabel('Predicted')

    total_plots = len(score_columns) + 1 
    for idx in range(total_plots, 6):
        fig.delaxes(ax.flatten()[idx])

    plt.tight_layout()
    plt.show()



def main(df):
    X, y, score_columns = preprocess_data(df)
    
    expected_scores = ['accuracy', 'fluency', 'prosodic', 'completeness']
    score_columns = [col for col in expected_scores if col in df.columns]
    
    print(f"Training with scores: {score_columns}")
    
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
        num_scores=len(score_columns), 
        max_length=800,
    )

    print("Model structure")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output scores: {score_columns}\n")

    train_losses, val_losses = training_model(
        model, train_loader, val_loader,
        num_epochs=50, learning_rate=0.001, patience=10,
    )

    model.load_state_dict(torch.load('best_pronunciation_model.pth'))
    predictions, targets,  = evaluate_model(
        model, test_loader, score_columns
    )

    results_dict = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'predictions': predictions,
        'targets': targets,
        'score_columns': score_columns,
    }
    plot_result(train_losses, val_losses, predictions, targets, score_columns)

    return model, predictions, targets, results_dict


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
    model, predictions, targets, results = main(combined_df)