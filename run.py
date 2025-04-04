import os
import pickle
import torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from src.utils import output_path
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc


class PatientDataset(Dataset):
    def __init__(self, data_folder):
        self.data = []
        self.labels = []
        self.ids = []

        for file_name in os.listdir(data_folder):
            if file_name.endswith('.pkl'):
                file_path = os.path.join(data_folder, file_name)
                with open(file_path, 'rb') as f:
                    vector = pickle.load(f)
                    self.data.append(vector[:-1])
                    self.labels.append(vector[-1])
                    self.ids.append(file_name.split('.')[0])

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.ids[idx]


# Model
class Model(nn.Module):
    def __init__(self, input_channels, hidden_channels, dropout):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Training
def train_model(model, dataloader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for features, labels, _ in dataloader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")


# Evaluate with PRAUC and ROC AUC
def evaluate_model(model, dataloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels, ids in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = correct / total
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
    pr_auc = auc(recall, precision)

    # ROC AUC
    roc_auc = roc_auc_score(all_labels, all_predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"PRAUC: {pr_auc:.4f}, ROC AUC: {roc_auc:.4f}")
    return accuracy, pr_auc, roc_auc


parser = argparse.ArgumentParser(description="Run prediction.")
parser.add_argument("--task", type=str, default="readmission")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.01)
args = parser.parse_args()

# train, val, test folders path
train_folder = os.path.join(output_path, args.task, "train")
val_folder = os.path.join(output_path, args.task, "val")
test_folder = os.path.join(output_path, args.task, "test")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Datasets and Dataloaders
train_dataset = PatientDataset(train_folder)
val_dataset = PatientDataset(val_folder)
test_dataset = PatientDataset(test_folder)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Model, Loss, Optimizer
model = Model(input_channels=len(train_dataset[0][0]), hidden_channels=128, dropout=args.dropout)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)

# Train and Evaluate
train_model(model, train_loader, criterion, optimizer, num_epochs=args.epochs, device=device)
torch.save(model.state_dict(), os.path.join(output_path, f"best_model_{args.task}.pt"))
print("Evaluating on the val set:")
evaluate_model(model, val_loader, device=device)

# Evaluate on the test set
print("Evaluating on the test set:")
evaluate_model(model, test_loader, device=device)

# load best model
# print("Load best model:")
# new_model = Model(input_channels=len(train_dataset[0][0]), hidden_channels=128, dropout=args.dropout)
# new_model.to(device)
# new_model.load_state_dict(torch.load(os.path.join(output_path, f"best_model_{args.task}.pt")))
# new_model.eval()
# evaluate_model(new_model, test_loader, device=device)
