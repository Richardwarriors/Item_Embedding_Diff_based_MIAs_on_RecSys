import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
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


# Evaluate Model and Return ROC Data
def get_roc_data(model, test_data, test_labels):
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        probabilities = F.softmax(outputs, dim=1)[:, 1].numpy()
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(test_labels, probabilities)
    return fpr, tpr


# Train Model and Return ROC Data
def train_and_get_roc(train_file, test_file):
    # Load train and test data
    train_data, train_labels = load_data(train_file)
    test_data, test_labels = load_data(test_file)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Initialize model, criterion, and optimizer
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    epochs = 100  # 降低训练轮次以加快运行速度
    for epoch in range(epochs):
        model.train()
        for i in range(len(train_data)):
            optimizer.zero_grad()
            output = model(train_data[i].unsqueeze(0))
            loss = criterion(output, train_labels[i].unsqueeze(0))
            loss.backward()
            optimizer.step()

    # Return ROC Data
    fpr, tpr = get_roc_data(model, test_data, test_labels)
    return fpr, tpr


# Plot ROC Curves for BL and DL
def plot_roc_curves_log(path, files):
    plt.figure(figsize=(8, 6))
    for char in files:
        train_file = f"{path}/trainForClassifier_{char}.txt"
        test_file = f"{path}/testForClassifier_{char}.txt"

        # Train model and get ROC data
        fpr, tpr = train_and_get_roc(train_file, test_file)

        # Plot the ROC curve
        plt.plot(fpr, tpr, label=f"{char} ROC Curve (AUC = {np.trapz(tpr, fpr):.4f})")

    # Customize log-scale plot
    plt.xscale('log')  # X轴设置为对数刻度
    plt.yscale('log')  # Y轴设置为对数刻度
    plt.xlim([1e-5, 1.0])  # 突出 FPR 范围
    plt.ylim([1e-5, 1.0])  # 突出 TPR 范围

    # 标注对数刻度
    plt.xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], ['10⁻⁵', '10⁻⁴', '10⁻³', '10⁻²', '10⁻¹', '1'])
    plt.yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], ['10⁻⁵', '10⁻⁴', '10⁻³', '10⁻²', '10⁻¹', '1'])

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves with Logarithmic Scales for MIA-RS')
    plt.legend(loc="lower right")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # 添加网格
    plt.show()


# Main Function
if __name__ == "__main__":
    path = "../Attackdata"  # 数据文件路径
    files = ['MN','DN','BN','ML','BL', 'DL']  # 文件标识符
    plot_roc_curves_log(path, files)

