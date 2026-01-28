import pandas as pd
from scipy.spatial.distance import cosine, braycurtis, cityblock, sqeuclidean
import math
import time

start_time = time.time()

train_member = pd.read_csv("../data/processed_AMB/target_member.csv")
member_interacted_dict = train_member.groupby('userId')['itemId'].apply(list).to_dict()

train_member_recommendation = pd.read_csv("../data/processed_AMB/processed_NAMB/target_recommendations.csv")
train_member_recommendation_dict = train_member_recommendation.set_index('raw_uid').apply(list, axis=1).to_dict()

item_matrix = pd.read_csv("../data/processed_AMB/mf_item_matrix.csv")
item_embedding_dict = item_matrix.set_index('raw_iid').iloc[:, 1:].apply(list, axis=1).to_dict()

train_data_member = {}
for user_id, item_list in member_interacted_dict.items():
    for item in item_list:
        train_data_member[(user_id, item)] = []
        rank = 1
        for rec_item in train_member_recommendation_dict[user_id]:
            s_cos = cosine(item_embedding_dict[item], item_embedding_dict[rec_item])
            s_bray = braycurtis(item_embedding_dict[item], item_embedding_dict[rec_item])
            s_city = cityblock(item_embedding_dict[item], item_embedding_dict[rec_item])
            s_sqe = sqeuclidean(item_embedding_dict[item], item_embedding_dict[rec_item])
            #s_cos = math.log2(rank + 1) * s_cos
            #s_bray = math.log2(rank + 1) * s_bray
            #s_city = math.log2(rank + 1) * s_city
            #s_sqe = math.log2(rank + 1) * s_sqe
            train_data_member[(user_id, item)].append(s_cos)
            train_data_member[(user_id, item)].append(s_bray)
            train_data_member[(user_id, item)].append(s_city)
            train_data_member[(user_id, item)].append(s_sqe)
            rank += 1

print('---------------------non-member------------------------')

train_nonmember = pd.read_csv("../data/processed_AMB/target_nonmember")
nonmember_interacted_dict = train_nonmember.groupby('userId')['itemId'].apply(list).to_dict()

train_nonmember_recommendation = pd.read_csv("../data/processed_AMB/processed_NAMB/target_nonmember_recommendation.csv")
train_nonmember_recommendation_dict = train_nonmember_recommendation.groupby('userId')['itemId'].apply(list).to_dict()

train_data_nonmember = {}
for user_id, item_list in nonmember_interacted_dict.items():
    for item in item_list:
        train_data_nonmember[(user_id, item)] = []
        rank = 1
        for rec_item in train_nonmember_recommendation_dict[user_id]:
            s_cos = cosine(item_embedding_dict[item], item_embedding_dict[rec_item])
            s_bray = braycurtis(item_embedding_dict[item], item_embedding_dict[rec_item])
            s_city = cityblock(item_embedding_dict[item], item_embedding_dict[rec_item])
            s_sqe = sqeuclidean(item_embedding_dict[item], item_embedding_dict[rec_item])
            train_data_nonmember[(user_id, item)].append(s_cos)
            train_data_nonmember[(user_id, item)].append(s_bray)
            train_data_nonmember[(user_id, item)].append(s_city)
            train_data_nonmember[(user_id, item)].append(s_sqe)
            rank += 1



#create attack file
dir_path = '../CIKM_Attack'
with open(dir_path + "/test_BN.txt", 'w') as f:
    for (user_id, item_id), feature_list in train_data_member.items():
        line = f"({user_id}, {item_id})\t{feature_list}\t[1]\n"
        f.write(line)
    for (user_id, item_id), feature_list in train_data_nonmember.items():
        line = f"({user_id}, {item_id})\t{feature_list}\t[0]\n"
        f.write(line)

end_time = time.time()
print(f"The time spent is: ",end_time - start_time)


