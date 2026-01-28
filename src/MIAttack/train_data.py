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
    vector = pd.read_csv(filename, sep=',')
    vector.columns = ['userId','itemId','rating','timestamp']
    if col == 'mem':
        vector['rating'] = 1
    else:
       	vector['rating'] = 0
    #print(ml_1m_vector.dtypes)
    return vector

def csv_to_dict(filename,data_dict):
    user_interactions = filename.groupby('userId')['itemId'].apply(list).to_dict()
    # 打印每个用户的互动Item列表
    for userid, items in user_interactions.items():
        #print(userid,items)
        data_dict[userid] = items
    return data_dict

topk = 100
num_latent = 100

path_shadow_interaction = "../data/processed_AMB"
path_shadow_recommend = "../data/processed_AMB/processed_LAMB"

f_shadow_mem = readData(path_shadow_interaction + "/shadow_member.csv",'mem') # interactions for shadow member
print("是否包含 NaN:", f_shadow_mem.isnull().values.any())
print("是否包含 Inf:", (f_shadow_mem == np.inf).values.any() or (f_shadow_mem == -np.inf).values.any())
f_shadow_nomem = readData(path_shadow_interaction + "/shadow_nonmember",'nonmem') #interactions for shadow  nonmember
print("是否包含 NaN:", f_shadow_nomem.isnull().values.any())
print("是否包含 Inf:", (f_shadow_nomem == np.inf).values.any() or (f_shadow_nomem == -np.inf).values.any())

fr_shadow_mem = pd.read_csv(path_shadow_recommend + "/shadow_recommendations.csv", sep = ',') # recommend for shadow  member
print("是否包含 NaN:", fr_shadow_mem.isnull().values.any())
print("是否包含 Inf:", (fr_shadow_mem == np.inf).values.any() or (fr_shadow_mem == -np.inf).values.any())
fr_shadow_nonmem = pd.read_csv(path_shadow_recommend + "/shadow_nonmember_recommendation.csv", sep = ',') # recommend for shadow  nonmembe
print("是否包含 NaN:", fr_shadow_nonmem.isnull().values.any())
print("是否包含 Inf:", (fr_shadow_nonmem == np.inf).values.any() or (fr_shadow_nonmem == -np.inf).values.any())
#fr_target_mem.replace([np.inf, -np.inf], np.nan, inplace=True)
#fr_target_mem.dropna(inplace=True)

fr_vector_shadow = pd.read_csv(path_shadow_interaction + "/mf_item_matrix.csv", sep = ',') # vector for shadow items
print("是否包含 NaN:", fr_vector_shadow.isnull().values.any())
print("是否包含 Inf:", (fr_vector_shadow == np.inf).values.any() or (fr_vector_shadow == -np.inf).values.any())

num_shadow_item = f_shadow_mem['itemId'].nunique()
print(f"the shadow member item has {num_shadow_item}")


num_shadow_mem = f_shadow_mem['userId'].nunique()
print(f"the shadow member user has {num_shadow_mem}")
num_shadow_nonmem = f_shadow_nomem['userId'].nunique()
print(f"the shadow nonmember user has {num_shadow_mem}")

num_vector_shadow = len(fr_vector_shadow)
print(f"the item vector has {num_vector_shadow}")

interaction_shadow_mem = {} # interactions for shadow  member
interaction_shadow_nonmem = {} # interactions for shadow nonmember

recommend_shadow_mem = {}   # recommends for shadow member
recommend_shadow_nonmem = {} # recommends for shadow nonmember

vector_shadow = {}  # vectors for shadow
label_shadow = {}


# vectors for shadow items
vectors_shadow = {fr_vector_shadow.iloc[i,1] : torch.tensor(fr_vector_shadow.iloc[i,2:].values.astype(float)) for i in range(num_vector_shadow)}
#for item, item_vector in vectors_shadow.items():
#   print(f"item is {item}, item_vector is {item_vector}")

# init for shadow

# read recommends shadow member
for i in range(len(fr_shadow_mem)):
    userid = fr_shadow_mem.iloc[i, 0]  # 获取 userid
    itemids = fr_shadow_mem.iloc[i, 1:].astype(int).tolist()  # 获取从第二列开始的所有 itemid，转换为整数
    recommend_shadow_mem[userid] = itemids  # 存入 recommend_target 中
#for userid, itemids in recommend_shadow_mem.items():
#   print(f"userid is {userid}, itemids is {itemids}")


#read recommends shadow nonmember
recommend_shadow_nonmem = csv_to_dict(fr_shadow_nonmem,recommend_shadow_nonmem)
#for userid, itemids in recommend_shadow_nonmem.items():
#   print(f"userid is {userid}, itemids is {itemids}")


#read interactions shadow mem
interaction_shadow_mem = csv_to_dict(f_shadow_mem,interaction_shadow_mem)
#for userid, itemids in interaction_shadow_mem.items():
#   print(f"userid is {userid}, itemids is {itemids}")

#read interactions shadow nonmem
interaction_shadow_nonmem = csv_to_dict(f_shadow_nomem,interaction_shadow_nonmem)
#for userid, itemids in interaction__shadow_nonmem.items():
#   print(f"userid is {userid}, itemids is {itemids}")


# vectorization for shadow mem
for userid in recommend_shadow_mem.keys():

    label_shadow[userid] = [1]

    interaction_vector = torch.zeros(100)
   
    #check whether userid is NAN
    if not pd.isna(userid):
        len_shadow = len(interaction_shadow_mem[userid])
    else:
        print(f"{userid} not exist in recommend_shadow")
        continue
   
    for j in range(len_shadow):
        if interaction_shadow_mem[userid][j] not in vectors_shadow:
            print(f"userid is {userid}")
            print(f"{interaction_shadow_mem[userid][j]} not happened in vector dataset")
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
            print(f"{recommend_shadow_mem[userid][j]} not happened in vector dataset")
            recommend_vector = recommend_vector + torch.zeros(100)
    #recommend_vector = recommend_vector.numpy().tolist()
    # subtracted
    vector_shadow[userid] = (interaction_vector - recommend_vector).numpy().tolist()
#for userid, vector_diff in vector_shadow.items():
#   print(f"userid is {userid}, vector_diff is {vector_diff}")

print('-' * 80)

## vectorization for shadow nonmem
for userid in recommend_shadow_nonmem.keys():

    label_shadow[userid] = [0]

    interaction_vector = torch.zeros(100)

    len_shadow = len(interaction_shadow_nonmem[userid])
    for j in range(len_shadow):
        if interaction_shadow_nonmem[userid][j] not in vectors_shadow:
            print(f"userid is {userid}")
            print(f"{interaction_shadow_nonmem[userid][j]} not happened in vector dataset")
            interaction_vector = interaction_vector + torch.zeros(100)
        else:
            interaction_vector = interaction_vector + vectors_shadow[interaction_shadow_nonmem[userid][j]]
    interaction_vector = interaction_vector / len_shadow
    #interaction_vector = interaction_vector.numpy().tolist()

    recommend_vector = torch.zeros(100)
    for j in range(topk):
        if recommend_shadow_nonmem[userid][j] in vectors_shadow:
            recommend_vector = recommend_vector + (1 / topk) * vectors_shadow[recommend_shadow_nonmem[userid][j]]
        else:
            print(f"{recommend_shadow_nonmem[userid][j]} not happened in vector dataset")
            recommend_vector = recommend_vector + torch.zeros(100)
    #recommend_vector = recommend_vector.numpy().tolist()
    # subtracted
    vector_shadow[userid] = (interaction_vector - recommend_vector).numpy().tolist()



#create test data for Attack model
dir_path = '../Attackdata'
ShadowDataset = open(dir_path + "/trainForClassifier_BL.txt", 'w')
user_shadow = list(recommend_shadow_mem.keys()) + list(recommend_shadow_nonmem.keys())

for userid in user_shadow:
    if userid in list(vector_shadow.keys()):
        ShadowDataset.write(str(vector_shadow[userid]) + '\t' + str(label_shadow[userid]) + '\n')
    else:
       print(f"{userid} not happend in vector_shadow")
    #vector_shadow[userid] = torch.Tensor(np.array(vector_shadow[userid])) # train
    #label_shadow[userid] = torch.Tensor(np.array(label_shadow[userid])).long() # train
