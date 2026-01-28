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
    #print(vector.dtypes)
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

path_target_interaction = "../data/processed_AMB"
path_target_recommend = "../data/processed_AMB/processed_LAMB"

f_target_mem = readData(path_target_interaction + "/target_member.csv",'mem') # interactions for target member
print("是否包含 NaN:", f_target_mem.isnull().values.any())
print("是否包含 Inf:", (f_target_mem == np.inf).values.any() or (f_target_mem == -np.inf).values.any())
f_target_nomem = readData(path_target_interaction + "/target_nonmember",'nonmem') #interactions for target nonmember
print("是否包含 NaN:", f_target_nomem.isnull().values.any())
print("是否包含 Inf:", (f_target_nomem == np.inf).values.any() or (f_target_nomem == -np.inf).values.any())

fr_target_mem = pd.read_csv(path_target_recommend + "/target_recommendations.csv", sep = ',') # recommend for target member
print("是否包含 NaN:", fr_target_mem.isnull().values.any())
print("是否包含 Inf:", (fr_target_mem == np.inf).values.any() or (fr_target_mem == -np.inf).values.any())
fr_target_nonmem = pd.read_csv(path_target_recommend + "/target_nonmember_recommendation.csv", sep = ',') # recommend for target nonmembe
print("是否包含 NaN:", fr_target_nonmem.isnull().values.any())
print("是否包含 Inf:", (fr_target_nonmem == np.inf).values.any() or (fr_target_nonmem == -np.inf).values.any())
#fr_target_mem.replace([np.inf, -np.inf], np.nan, inplace=True)
#fr_target_mem.dropna(inplace=True)

fr_vector_target = pd.read_csv(path_target_interaction + "/mf_item_matrix.csv", sep = ',') # vector for target items
print("是否包含 NaN:", fr_vector_target.isnull().values.any())
print("是否包含 Inf:", (fr_vector_target == np.inf).values.any() or (fr_vector_target == -np.inf).values.any())

num_target_item = f_target_mem['itemId'].nunique()
print(f"the target member item has {num_target_item}")


num_target_mem = f_target_mem['userId'].nunique()
print(f"the target member user has {num_target_mem}")
num_target_nonmem = f_target_nomem['userId'].nunique()
print(f"the target nonmember user has {num_target_mem}")

num_vector_target = len(fr_vector_target)
print(f"the item vector has {num_vector_target}")

interaction_target_mem = {} # interactions for target member
interaction_target_nonmem = {} # interactions for target nonmember

recommend_target_mem = {}   # recommends for target member
recommend_target_nonmem = {} # recommends for target nonmember

vector_target = {}  # vectors for target
label_target = {}


# vectors for target items
vectors_target = {fr_vector_target.iloc[i,1] : torch.tensor(fr_vector_target.iloc[i,2:].values.astype(float)) for i in range(num_vector_target)}
#for item, item_vector in vectors_target.items():
#   print(f"item is {item}, item_vector is {item_vector}")

# init for target

# read recommends target member
#print(fr_target_mem.dtypes())
for i in range(len(fr_target_mem)):
    userid = fr_target_mem.iloc[i, 0]  # 获取 userid
    itemids = fr_target_mem.iloc[i, 1:].astype(int).tolist()  # 获取从第二列开始的所有 itemid，转换为整数
    recommend_target_mem[userid] = itemids  # 存入 recommend_target 中
#for userid, itemids in recommend_target_mem.items():
#   print(f"userid is {userid}, itemids is {itemids}")


#read recommends target nonmember
recommend_target_nonmem = csv_to_dict(fr_target_nonmem,recommend_target_nonmem)
#for userid, itemids in recommend_target_nonmem.items():
#   print(f"userid is {userid}, itemids is {itemids}")


#read interactions target mem
interaction_target_mem = csv_to_dict(f_target_mem,interaction_target_mem)
#for userid, itemids in interaction_target_mem.items():
#   print(f"userid is {userid}, itemids is {itemids}")

#read interactions target nonmem
interaction_target_nonmem = csv_to_dict(f_target_nomem,interaction_target_nonmem)
#for userid, itemids in interaction__target_nonmem.items():
#   print(f"userid is {userid}, itemids is {itemids}")


# vectorization for target mem
for userid in recommend_target_mem.keys():

    label_target[userid] = [1]

    interaction_vector = torch.zeros(100)
   
    #check whether userid is NAN
    if not pd.isna(userid):
        len_target = len(interaction_target_mem[userid])
    else:
        print(f"{userid} not exist in recommend_target")
        continue
   
    for j in range(len_target):
        if interaction_target_mem[userid][j] not in vectors_target:
            print(f"userid is {userid}")
            print(f"{interaction_target_mem[userid][j]} not happened in vector dataset")
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
            print(f"{recommend_target_mem[userid][j]} not happened in vector dataset")
            recommend_vector = recommend_vector + torch.zeros(100)
    #recommend_vector = recommend_vector.numpy().tolist()
    # subtracted
    vector_target[userid] = (interaction_vector - recommend_vector).numpy().tolist()
#for userid, vector_diff in vector_target.items():
#   print(f"userid is {userid}, vector_diff is {vector_diff}")

print('-' * 80)

## vectorization for target nonmem
for userid in recommend_target_nonmem.keys():

    label_target[userid] = [0]

    interaction_vector = torch.zeros(100)

    len_target = len(interaction_target_nonmem[userid])
    for j in range(len_target):
        if interaction_target_nonmem[userid][j] not in vectors_target:
            print(f"userid is {userid}")
            print(f"{interaction_target_nonmem[userid][j]} not happened in vector dataset")
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
            print(f"{recommend_target_nonmem[userid][j]} not happened in vector dataset")
            recommend_vector = recommend_vector + torch.zeros(100)
    #recommend_vector = recommend_vector.numpy().tolist()
    # subtracted
    vector_target[userid] = (interaction_vector - recommend_vector).numpy().tolist()



#create test data for Attack model
dir_path = '../Attackdata'
TargetDataset = open(dir_path + "/testForClassifier_BL.txt", 'w')
user_target = list(recommend_target_mem.keys()) + list(recommend_target_nonmem.keys())
#print(user_target)
for userid in user_target:
    if userid in list(vector_target.keys()):
        TargetDataset.write(str(vector_target[userid]) + '\t' + str(label_target[userid]) + '\n')
    else:
       print(f"{userid} not happend in vector_target")
    #vector_target[userid] = torch.Tensor(np.array(vector_target[userid])) # test
    #label_target[userid] = torch.Tensor(np.array(label_target[userid])).long() # test
