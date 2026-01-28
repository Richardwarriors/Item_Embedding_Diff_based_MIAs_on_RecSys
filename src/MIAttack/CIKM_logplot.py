import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve, roc_auc_score


# MLP模型定义
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(400, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)
        # self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x

def load_data(filename):
    vectors = []
    labels = []
    user_item_pairs = []
    with open(filename, 'r') as file:
        for line in file:
            user_item_pair,vector_str, label_str = line.strip().split('\t')
            user_item_pair = eval(user_item_pair)
            vector = eval(vector_str)
            label = int(eval(label_str)[0])
            user_item_pairs.append(user_item_pair)
            vectors.append(vector)
            labels.append(label)
    return np.array(user_item_pairs), np.array(vectors), np.array(labels)


# 模型训练并获取 FPR 和 TPR
def train_and_get_roc(train_file, test_file):
    # 加载数据
    train_data, train_labels = load_data(train_file)
    test_data, test_labels = load_data(test_file)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # 初始化模型
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 模型训练
    epochs = 100  # 降低训练轮次加速运行
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

    # 评估模型，计算 FPR 和 TPR
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        probabilities = F.softmax(outputs, dim=1)[:, 1].numpy()
        fpr, tpr, _ = roc_curve(test_labels, probabilities)
    return fpr, tpr


LiRA_data = {
    "MN": [0.392, 0.525, 0.671, 0.919,1],
    "ML": [0.378, 0.496, 0.678, 0.921,1],
    "DN": [0.572, 0.589, 0.921, 0.987,1],
    "DL": [0.568, 0.692, 0.981, 0.992,1],
    "BN": [0.331, 0.357, 0.989, 0.993,1],
    "BL": [0.363, 0.653, 0.991, 0.992,1]
}



# 绘制 FPR-TPR 图像（对数刻度）
def plot_log_roc_curves(path, files):
    plt.figure(figsize=(8, 6))

    for char in files:
        train_file = f"{path}/train_{char}.txt"
        test_file = f"{path}/test_{char}.txt"
        label = f"{char} CIKM"
        # 获取 FPR 和 TPR
        fpr, tpr = train_and_get_roc(train_file, test_file)

        # 绘制曲线
        plt.plot(fpr, tpr, label = label)

    for key, tpr_values in LiRA_data.items():
        fpr_values = [0.0001, 0.001, 0.01, 0.1, 1.0]  # 固定的 FPR 值
        plt.plot(fpr_values, tpr_values, label=f"{key} RecPS", linestyle='--')
    # 设置对数刻度
    plt.xscale('log')  # X轴对数
    plt.yscale('log')  # Y轴对数
    plt.xlim([1e-4, 1.0])  # 限定 X 轴范围
    plt.ylim([1e-4, 1.0])  # 限定 Y 轴范围

    # 自定义刻度，突出 10⁻⁵ 到 1
    plt.xticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0], ['10⁻⁴', '10⁻³', '10⁻²', '10⁻¹', '1'])
    plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0], ['10⁻⁴', '10⁻³', '10⁻²', '10⁻¹', '1'])

    # 图像设置
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    #plt.title('Comparing the true-positive rate vs. false-positive rate for RecPS and item-diff')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # 添加网格
    plt.legend(loc="lower right")
    plt.savefig("tpr-fpr.png", dpi=300, bbox_inches='tight')
    plt.show()


# 主函数
if __name__ == "__main__":
    path = "../Attackdata"  # 数据文件路径
    files = ['MN','DN','BN','ML','BL', 'DL']  # 文件标识符
    plot_log_roc_curves(path, files)
