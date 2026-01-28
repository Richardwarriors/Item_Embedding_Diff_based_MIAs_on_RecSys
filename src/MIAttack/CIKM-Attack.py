import pandas as pd
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, roc_auc_score


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
        x = F.softmax(self.fc3(x), dim=1)
        return x


# 使用ROC曲线评估模型
def evaluate_model_with_roc_curve(model, test_data, test_labels, target_fpr):
    # 获取模型输出概率
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        probabilities = F.softmax(outputs, dim=1)[:, 1].numpy()
    print("Output probabilities:", probabilities[:10])

    # 使用roc_curve计算fpr、tpr和阈值
    fpr, tpr, thresholds = roc_curve(test_labels, probabilities)
    print("FPR values:", fpr[:10])
    # 找到目标FPR对应的阈值
    idx = np.argmax(fpr >= target_fpr)

    # 如果满足目标FPR的阈值在范围内，则选取，否则返回最后一个可用的
    best_threshold = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
    best_tpr = tpr[idx]
    best_fpr = fpr[idx]

    global_auc = roc_auc_score(test_labels, probabilities)

    print(f'目标FPR: {target_fpr}, 实际FPR: {best_fpr:.4f}, TPR: {best_tpr:.4f}, 阈值: {best_threshold:.4f}')
    print(f'全局AUC: {global_auc:.4f}')

    return best_threshold, best_tpr, best_fpr, global_auc


# 数据加载
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


# 模型训练
def train_model(train_file, test_file, target_fpr):
    # 加载训练和测试数据
    train_user_item,train_data, train_labels = load_data(train_file)
    test_user_item,test_data, test_labels = load_data(test_file)
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # 初始化MLP模型
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 模型训练
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        # 每个epoch后在训练集上评估AUC
        with torch.no_grad():
            outputs = model(train_data)
            auc = roc_auc_score(train_labels.numpy(), F.softmax(outputs, dim=1)[:, 1].numpy())
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Train AUC: {auc:.4f}')

    # 评估模型在测试集上的表现，寻找满足目标FPR的最佳阈值
    with torch.no_grad():
        best_threshold, best_tpr, best_fpr, global_auc = evaluate_model_with_roc_curve(model, test_data, test_labels,
                                                                                       target_fpr)
        return best_threshold, best_tpr, best_fpr, global_auc


path = "../CIKM_Attack"
# 数据路径和文件
#for char in ['MN']:
#    train_file = path + "/train_" + char + ".txt"
#    test_file = path + "/test_" + char + ".txt"
#    for target_fpr in [0.0001, 0.001, 0.01, 0.1]:
#        best_threshold, best_tpr, best_fpr, global_auc = train_model(train_file, test_file, target_fpr)
#        print(f'在目标FPR下最佳TPR: {best_tpr}, 全局AUC: {global_auc}')

train_file = path + "/train_DN.txt"
test_file = path + "/test_DN.txt"
#for target_fpr in [0.0001, 0.001, 0.01, 0.1]:
#    best_threshold, best_tpr, best_fpr, global_auc = train_model(train_file, test_file, target_fpr)
#    print(f'在目标FPR下最佳TPR: {best_tpr}, 全局AUC: {global_auc}')
train_user_item,train_data, train_labels = load_data(train_file)

unique, counts = np.unique(train_labels, return_counts=True)
print(dict(zip(unique, counts)))
