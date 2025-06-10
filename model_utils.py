import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import copy
import time
import os
from tqdm import tqdm

class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"ERROR: Image not found at {img_path}. Check unzipping and paths.")
            # This can happen if image_paths list is not perfectly in sync with actual files
            # or if MAX_IMAGES_TO_LOAD caused a mismatch.
            # Returning None, None will be handled by a custom collate_fn or checked in the loop.
            return None, None
        except Exception as e:
            print(f"ERROR: Could not open image {img_path}. Error: {e}")
            return None, None

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label.unsqueeze(0) # Make label [1] for BCELoss

class SimpleCNN(nn.Module):
    def __init__(self, img_width, img_height, num_classes=1):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 64x64 -> 32x32

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # pool 32x32 -> 16x16

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # pool 16x16 -> 8x8

        # Dynamically calculate the flattened size
        # This makes the model more robust to changes in IMG_WIDTH/IMG_HEIGHT or pooling layers
        with torch.no_grad(): # No need to track gradients for this dummy pass
            # dummy input tensor with the expected shape (batch_size=1, channels=3, H, W)
            dummy_input = torch.zeros(1, 3, img_height, img_width)
            # go through the convolutional and pooling layers
            dummy_output = self._forward_features(dummy_input)
            self.flattened_size = dummy_output.numel() // dummy_output.shape[0] # numel() gives total elements

        # fully connectedf layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def _forward_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(-1, self.flattened_size) # Flatten
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x) # output logits
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, patience_early_stopping, model_save_path='best_model.pth'):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_model_wts = None
    epochs_no_improve = 0
    start_time_total = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        running_loss_train = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]", leave=False):
            if inputs is None or labels is None:
                continue
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss_train += loss.item() * inputs.size(0)
            predicted = torch.sigmoid(outputs) > 0.5
            total_train += labels.size(0)
            correct_train += (predicted == labels.bool()).sum().item()
            
        epoch_train_loss = running_loss_train / total_train if total_train > 0 else 0
        epoch_train_acc = correct_train / total_train if total_train > 0 else 0
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # --- Validation Phase ---
        model.eval()
        running_loss_val = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VALID]", leave=False):
                if inputs is None or labels is None:
                    continue
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss_val += loss.item() * inputs.size(0)
                predicted = torch.sigmoid(outputs) > 0.5
                total_val += labels.size(0)
                correct_val += (predicted == labels.bool()).sum().item()

        epoch_val_loss = running_loss_val / total_val if total_val > 0 else 0
        epoch_val_acc = correct_val / total_val if total_val > 0 else 0
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Time: {epoch_time:.2f}s | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
                print(f"Validation loss improved. Saved best model to '{model_save_path}'")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience_early_stopping:
            print(f"Early stopping triggered after {patience_early_stopping} epochs without improvement.")
            break
            
    total_training_time = time.time() - start_time_total
    print(f"\nTtraining finished in {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s.")

    if best_model_wts:
        print("Loading best model weights for final use.")
        model.load_state_dict(best_model_wts)
    elif os.path.exists(model_save_path):
        print(f"Loading model from last saved state at '{model_save_path}'.")
        model.load_state_dict(torch.load(model_save_path, map_location=device))

    return model, history

def plot_training_history(history):
    if not history or not history.get('train_acc'):
        print("History is empty or incomplete, cannot plot.")
        return

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def evaluate_model(model, data_loader, criterion, device):
    model.eval() # ev mode
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    all_preds_probs = [] #  probabilities for roc auc
    all_true_labels = []

    with torch.no_grad(): # Disable gradient calculations
        for inputs, labels in data_loader:
            if inputs is None or labels is None: # Skip pb data
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) # Get logits
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            predicted_probs = torch.sigmoid(outputs) # vonvert logits to probabilities
            predicted_classes = predicted_probs > 0.5 # ythreshold probabilities

            total_samples += labels.size(0)
            correct_preds += (predicted_classes == labels.bool()).sum().item()

            all_preds_probs.extend(predicted_probs.cpu().numpy().flatten())
            all_true_labels.extend(labels.cpu().numpy().astype(int).flatten())


    avg_loss = running_loss / total_samples if total_samples > 0 else float('nan')
    accuracy = correct_preds / total_samples if total_samples > 0 else float('nan')

    return avg_loss, accuracy, all_preds_probs, all_true_labels