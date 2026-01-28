import pandas as pd
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve, roc_auc_score


# MLP Model Definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(100, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        #x = x.view(-1, 100)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


# Load Data
def load_data(filename):
    vectors = []
    labels = []
    with open(filename, 'r') as file:
        for line in file:
            vector_str, label_str = line.strip().split('\t')
            vector = eval(vector_str)
            label = int(eval(label_str)[0])
            vectors.append(vector)
            labels.append(label)
    return np.array(vectors), np.array(labels)


# Evaluate Model with ROC Curve and Low FPR
def evaluate_low_fpr(model, test_data, test_labels, target_fpr=0.001):
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        probabilities = F.softmax(outputs, dim=1)[:, 1].numpy()

    print("Output probabilities:", probabilities[:10])

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(test_labels, probabilities)
    print("FPR values:", fpr[:10])

    # Find TPR at target FPR
    idx = np.argmax(fpr >= target_fpr)
    best_tpr = tpr[idx] if idx < len(tpr) else tpr[-1]
    best_fpr = fpr[idx] if idx < len(fpr) else fpr[-1]

    global_auc = roc_auc_score(test_labels, probabilities)
    print(f"Target FPR: {target_fpr}, Actual FPR: {best_fpr:.6f}, TPR: {best_tpr:.6f}, Global AUC: {global_auc:.4f}")
    return best_tpr, global_auc


# Train Model
def train_model(train_file, test_file, target_fpr):
    # Load train and test data
    train_data, train_labels = load_data(train_file)
    test_data, test_labels = load_data(test_file)

    # Convert to tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Initialize model, criterion, and optimizer
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i in range(len(train_data)):
            optimizer.zero_grad()
            output = model(train_data[i].unsqueeze(0))
            loss = criterion(output, train_labels[i].unsqueeze(0))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Compute train AUC
        with torch.no_grad():
            all_outputs = [model(train_data[i].unsqueeze(0))[:, 1].item() for i in range(len(train_data))]
            train_auc = roc_auc_score(train_labels.numpy(), np.array(all_outputs))
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss / len(train_data):.4f}, Train AUC: {train_auc:.4f}")

    # Evaluate TPR at low FPR on test data
    best_tpr, global_auc = evaluate_low_fpr(model, test_data, test_labels, target_fpr)
    print(f"Best TPR at Target FPR ({target_fpr}): {best_tpr:.6f}, Global AUC: {global_auc:.4f}")


# Paths and Parameters
path = "../Attackdata"
train_file = path + "/trainForClassifier_BL.txt"
test_file = path + "/testForClassifier_BL.txt"
target_fpr = [0.0001,0.001,0.01,0.1]
for target_fpr in target_fpr:
    train_model(train_file, test_file, target_fpr)