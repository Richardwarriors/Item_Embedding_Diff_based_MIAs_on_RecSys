import pandas as pd
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score

def readData(filename,col):
    """
    读取数据，并将显式反馈转为隐式反馈
    """
    ml_1m_vector = pd.read_csv(filename, sep=',')
    ml_1m_vector.columns = ['userId','itemId','rating','timestamp']
    if col == 'mem':
        ml_1m_vector['rating'] = 1
    else:
        ml_1m_vector['rating'] = 0
    #print(ml_1m_vector.dtypes)
    return ml_1m_vector

def csv_to_dict(filename,data_dict):
    user_interactions = filename.groupby('userId')['itemId'].apply(list).to_dict()
    # 打印每个用户的互动Item列表
    for userid, items in user_interactions.items():
        #print(userid,items)
        data_dict[userid] = items
    return data_dict

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

topk = 100
num_latent = 100

path = "../data/processed_AMD"

f_target_mem = readData(path + "/target_member.csv",'mem') # interactions for target member
f_target_nomem = readData(path + "/target_nonmember",'nonmem') #interactions for target nonmember


fr_target_mem = pd.read_csv(path+ "/target_recommendations.csv", sep = ',') # recommend for target member
fr_target_nonmem = pd.read_csv(path+ "/target_nonmember_recommendation.csv", sep = ',') # recommend for target nonmembe
fr_target_mem.replace([np.inf, -np.inf], np.nan, inplace=True)
fr_target_mem.dropna(inplace=True)

fr_vector_target = pd.read_csv(path + "/item_matrix.csv", sep = ',') # vector for target items

num_target_mem = f_target_mem['userId'].nunique()
num_target_nonmem = f_target_nomem['userId'].nunique()

num_vector_target = len(fr_vector_target)
#print(num_vector_shadow, num_vector_target)

interaction_target_mem = {} # interactions for target member
interaction_target_nonmem = {} # interactions for target nonmember

recommend_target_mem = {}   # recommends for target member
recommend_target_nonmem = {} # recommends for target nonmember

vector_target = {}  # vectors for target
label_target = {}



# vectors for target items
vectors_target = {fr_vector_target.iloc[i,0] : torch.tensor(fr_vector_target.iloc[i,1:].values.astype(float)) for i in range(num_vector_target)}

# init for target

# read recommends target member
#print(fr_target_mem.dtypes())
for i in range(len(fr_target_mem)):
    userid = fr_target_mem.iloc[i, 0]  # 获取 userid
    itemids = fr_target_mem.iloc[i, 1:].astype(int).tolist()  # 获取从第二列开始的所有 itemid，转换为整数
    recommend_target_mem[userid] = itemids  # 存入 recommend_target 中
#print(recommend_target_mem)

#read recommends target nonmember
recommend_target_nonmem = csv_to_dict(fr_target_nonmem,recommend_target_nonmem)

# read interactions target mem
# 根据 UserID 分组并将 ItemID 生成列表
interaction_target_mem = csv_to_dict(f_target_mem,interaction_target_mem)
#print(interaction_target_mem)
#user_interactions = f_target_mem.groupby('userid')['itemid'].apply(list).to_dict()

# 打印每个用户的互动Item列表
#for userid, items in user_interactions.items():
#    interaction_target_mem[userid] = items
# read interactions target nonmem
interaction_target_nonmem = csv_to_dict(f_target_nomem,interaction_target_nonmem)
# vectorization for target mem
for userid in recommend_target_mem.keys():

    label_target[userid] = [1]

    interaction_vector = torch.zeros(100)
    # the center of the ineractions
    #print(interaction_target_mem[userid])
    if not pd.isna(userid):
        len_target = len(interaction_target_mem[userid])
    else:
        print(userid)
        continue
    for j in range(len_target):
        if interaction_target_mem[userid][j] not in vectors_target:
            interaction_vector = interaction_vector + torch.zeros(100)
        else:
            interaction_vector = interaction_vector + vectors_target[interaction_target_mem[userid][j]]
    interaction_vector = interaction_vector / len_target
    #interaction_vector = interaction_vector.numpy().tolist()

    recommend_vector = torch.zeros(100)
    #print(recommend_target_mem[userid])
    for j in range(topk):
        if recommend_target_mem[userid][j] in vectors_target:
            recommend_vector = recommend_vector + (1 / topk) * vectors_target[recommend_target_mem[userid][j]]
        else:
            recommend_vector = recommend_vector + torch.zeros(100)
    #recommend_vector = recommend_vector.numpy().tolist()
    # subtracted
    vector_target[userid] = (interaction_vector - recommend_vector).numpy().tolist()

## vectorization for target nonmem
for userid in recommend_target_nonmem.keys():

    label_target[userid] = [0]

    interaction_vector = torch.zeros(100)
    # the center of the ineractions
    #print(interaction_target_mem[userid])
    len_target = len(interaction_target_nonmem[userid])
    for j in range(len_target):
        if interaction_target_nonmem[userid][j] not in vectors_target:
            interaction_vector = interaction_vector + torch.zeros(100)
        else:
            interaction_vector = interaction_vector + vectors_target[interaction_target_nonmem[userid][j]]
    interaction_vector = interaction_vector / len_target
    #interaction_vector = interaction_vector.numpy().tolist()

    recommend_vector = torch.zeros(100)
    #print(recommend_target_mem[userid])
    for j in range(topk):
        if recommend_target_nonmem[userid][j] in vectors_target:
            recommend_vector = recommend_vector + (1 / topk) * vectors_target[recommend_target_nonmem[userid][j]]
        else:
            recommend_vector = recommend_vector + torch.zeros(100)
    #recommend_vector = recommend_vector.numpy().tolist()
    # subtracted
    vector_target[userid] = (interaction_vector - recommend_vector).numpy().tolist()

#---------------------------------------------------------------shadow---------------------------------------------------
# vectors for shadow items
interaction_shadow_mem = {} # interactions for shadow member
interaction_shadow_nonmem = {} # interactions for shadow nonmember

recommend_shadow_mem = {}   # recommends for shadow member
recommend_shadow_nonmem = {} # recommends for shadow nonmember

vector_shadow = {}  # vectors for shadow
label_shadow = {}

f_shadow_mem = readData(path + "/shadow_member.csv",'mem') # interactions for target member
f_shadow_nomem = readData(path + "/shadow_nonmember",'nonmem') #interactions for target nonmember

fr_shadow_mem = pd.read_csv(path+ "/shadow_recommendations.csv", sep = ',') # recommend for target member
fr_shadow_nonmem = pd.read_csv(path+ "/shadow_nonmember_recommendation.csv", sep = ',')# recommend for target nonmember

fr_vector_shadow = pd.read_csv(path + "/item_matrix.csv", sep = ',') # vector for shadow items

num_vector_shadow = len(fr_vector_shadow)
vectors_shadow = {fr_vector_shadow.iloc[i,0] : torch.tensor(fr_vector_shadow.iloc[i,1:].values.astype(float)) for i in range(num_vector_shadow)}
#print(vectors_shadow)

# init for shadow

# read recommends shadow member
for i in range(len(fr_shadow_mem)):
    userid = fr_shadow_mem.iloc[i, 0]  # 获取 userid
    itemids = fr_shadow_mem.iloc[i, 1:].astype(int).tolist()  # 获取从第二列开始的所有 itemid，转换为整数
    recommend_shadow_mem[userid] = itemids  # 存入 recommend_target 中

#read recommends shadow nonmember
recommend_shadow_nonmem = csv_to_dict(fr_shadow_nonmem,recommend_shadow_nonmem)

# read interactions shadow mem
interaction_shadow_mem = csv_to_dict(f_shadow_mem,interaction_shadow_mem)
# read interactions shadow nonmem
interaction_shadow_nonmem = csv_to_dict(f_shadow_nomem,interaction_shadow_nonmem)
# vectorization for target mem
for userid in recommend_shadow_mem.keys():

    label_shadow[userid] = [1]

    interaction_vector = torch.zeros(100)
    # the center of the ineractions
    #print(interaction_target_mem[userid])
    if not pd.isna(userid):
        len_shadow = len(interaction_shadow_mem[userid])
    else:
        print(userid)
        continue    
    #len_shadow = len(interaction_shadow_mem[userid])
    for j in range(len_shadow):
        if interaction_shadow_mem[userid][j] not in vectors_shadow:
            interaction_vector = interaction_vector + torch.zeros(100)
        else:
            interaction_vector = interaction_vector + vectors_shadow[interaction_shadow_mem[userid][j]]
    interaction_vector = interaction_vector / len_shadow
    #interaction_vector = interaction_vector.numpy().tolist()

    recommend_vector = torch.zeros(100)
    #print(recommend_target_mem[userid])
    for j in range(topk):
        if recommend_shadow_mem[userid][j] in vectors_shadow:
            recommend_vector = recommend_vector + (1 / topk) * vectors_shadow[recommend_shadow_mem[userid][j]]
        else:
            recommend_vector = recommend_vector + torch.zeros(100)
    #recommend_vector = recommend_vector.numpy().tolist()
    # subtracted
    vector_shadow[userid] = (interaction_vector - recommend_vector).numpy().tolist()

## vectorization for target nonmem
for userid in recommend_shadow_nonmem.keys():

    label_shadow[userid] = [0]

    interaction_vector = torch.zeros(100)
    # the center of the ineractions
    #print(interaction_target_mem[userid])
    len_shadow = len(interaction_shadow_nonmem[userid])
    for j in range(len_shadow):
        if interaction_shadow_nonmem[userid][j] not in vectors_shadow:
            interaction_vector = interaction_vector + torch.zeros(100)
        else:
            interaction_vector = interaction_vector + vectors_shadow[interaction_shadow_nonmem[userid][j]]
    interaction_vector = interaction_vector / len_shadow
    #interaction_vector = interaction_vector.numpy().tolist()

    recommend_vector = torch.zeros(100)
    #print(recommend_target_mem[userid])
    for j in range(topk):
        if recommend_shadow_nonmem[userid][j] in vectors_shadow:
            recommend_vector = recommend_vector + (1 / topk) * vectors_shadow[recommend_shadow_nonmem[userid][j]]
        else:
            recommend_vector = recommend_vector + torch.zeros(100)
    #recommend_vector = recommend_vector.numpy().tolist()
    # subtracted
    vector_shadow[userid] = (interaction_vector - recommend_vector).numpy().tolist()



ShadowDataset = open(path + "/trainForClassifier_DNDN.txt", 'w')
TargetDataset = open(path + "/testForClassifier_DNDN.txt", 'w')
user_shadow = list(recommend_shadow_mem.keys()) + list(recommend_shadow_nonmem.keys())
#print(user_shadow)
for userid in user_shadow:
    if not pd.isna(userid):
        ShadowDataset.write(str(vector_shadow[userid]) + '\t' + str(label_shadow[userid]) + '\n')
    #vector_shadow[userid] = torch.Tensor(np.array(vector_shadow[userid])) # train
    #label_shadow[userid] = torch.Tensor(np.array(label_shadow[userid])).long() # train

user_target = list(recommend_target_mem.keys()) + list(recommend_target_nonmem.keys())
#print(user_target)
for userid in user_target:
    if not pd.isna(userid):
        TargetDataset.write(str(vector_target[userid]) + '\t' + str(label_target[userid]) + '\n')
    #vector_target[userid] = torch.Tensor(np.array(vector_target[userid])) # test
    #label_target[userid] = torch.Tensor(np.array(label_target[userid])).long() # test


#----------------------------------------attack---------------------------------------------
# 读取数据文件
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

# 训练模型并调优AUC
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
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 开始训练
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        outputs = model(train_data)

        # 计算损失
        loss = criterion(outputs, train_labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 评估训练集上的表现
        model.eval()
        with torch.no_grad():
            # 计算预测的概率
            outputs = model(train_data)
            _, predicted = torch.max(outputs, 1)

            # 计算AUC
            auc = roc_auc_score(train_labels, outputs[:, 1].numpy())

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Train AUC: {auc:.4f}')

    # 测试模型在测试集上的AUC表现
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data)
        test_auc = roc_auc_score(test_labels, test_outputs[:, 1].numpy())
        print(f'Test AUC: {test_auc:.4f}')

train_file = path + "/trainForClassifier_MLLG.txt"
test_file = path + "/testForClassifier_MLLG.txt"
train_model(train_file, test_file)

