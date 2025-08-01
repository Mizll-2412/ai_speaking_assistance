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
from preprocess_score_model import AudioProcessor
from score_model import AudioScoringDataset, PronunciationScoringModel

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
    X, y, score_columns, word_counts = AudioProcessor.preprocess_data(df)

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