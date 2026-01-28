import pandas as pd
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(100, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = x.view(-1, 100)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

topk = 100
num_latent = 100

path = "../Attackdata"

def load_data(filename):
    vectors = []
    labels = []
    with open(filename, 'r') as file:
        for line in file:
            vector_str, label_str = line.strip().split('\t')
            vector = eval(vector_str)  # 将字符串解析为列表
            label = int(eval(label_str)[0])  # 获取标签
            vectors.append(vector)
            labels.append(label)
    return np.array(vectors), np.array(labels)

def train_model(train_file, test_file):
    # 加载训练和测试数据
    train_data, train_labels = load_data(train_file)
    test_data, test_labels = load_data(test_file)

    # 转换数据为Tensor
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # 实例化模型、损失函数和优化器
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 开始逐个样本训练
    epochs = 100
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0

        for i in range(len(train_data)):  # 逐个样本进行训练
            optimizer.zero_grad()

            # 获取单个样本数据和标签
            input_sample = train_data[i].unsqueeze(0)  # 保持输入形状为 (1, 100)
            label_sample = train_labels[i].unsqueeze(0)  # 保持标签形状为 (1,)

            # 前向传播
            output = model(input_sample)

            # 计算损失
            loss = criterion(output, label_sample)
            train_loss += loss.item()

            # 反向传播
            loss.backward()
            optimizer.step()

            # 计算训练准确率
            _, predicted = torch.max(output, 1)
            if predicted == label_sample:
                train_correct += 1

        avg_loss = train_loss / len(train_data)
        accuracy = train_correct / len(train_data)

        # 计算 AUC
        model.eval()
        with torch.no_grad():
            all_outputs = []
            for i in range(len(train_data)):
                output = model(train_data[i].unsqueeze(0))
                all_outputs.append(output[:, 1].item())
            auc = roc_auc_score(train_labels.numpy(), np.array(all_outputs))

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Train AUC: {auc:.4f}")

    # 测试模型在测试集上的AUC表现
    model.eval()
    with torch.no_grad():
        all_outputs = []
        for i in range(len(test_data)):  # 逐个样本计算测试结果
            output = model(test_data[i].unsqueeze(0))
            all_outputs.append(output[:, 1].item())

        test_auc = roc_auc_score(test_labels.numpy(), np.array(all_outputs))
        print(f"Test AUC: {test_auc:.4f}")

train_file = path + "/trainForClassifier_MN.txt"
test_file = path + "/testForClassifier_MN.txt"
train_model(train_file, test_file)
