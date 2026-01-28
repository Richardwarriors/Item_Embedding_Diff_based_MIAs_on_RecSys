import numpy as np
import pandas as pd
import random

numOfRecommend = 100


def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)


# recommend for nonmember
def recommend(root_path, filename, output_dir):
    set_seed(2024)
    rec = []
    data = pd.read_csv(root_path + filename, sep=',')
    print(data.columns)

    all_items = set(data['itemId'].unique())
    user_data = data.groupby('userId')

    for user, value in user_data:
        interacted_items = set(value['itemId'].tolist())
        available_items = list(all_items - interacted_items)

        if len(available_items) >= numOfRecommend:
            recommended_items = random.sample(available_items, numOfRecommend)
        else:
            recommended_items = available_items  # 如果不足100个则全部推荐

        for item in recommended_items:
            rec.append({'userId': str(user), 'itemId': str(item), 'rating': '0'})

    rec_df = pd.DataFrame(rec)
    rec_df.to_csv(output_dir + filename[:16] + '_recommendation.csv', index=False)


if __name__ == '__main__':
    root_path = "../data/processed_ml-1m/"
    output_dir = "../data/processed_ml-1m/processed_Nml1m/"
    datalist = ["target_nonmember", "shadow_nonmember"]

    for filename in datalist:
        recommend(root_path, filename, output_dir)











'''
numOfRecommend = 100


def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)

# recommend for nonmember
def recommend(root_path, filename, output_dir):
    rec = []
    data = pd.read_csv(root_path + filename, sep=',')
    print(data.columns)
    Popularity = data.groupby('itemId').size()
    sorted_Popularity = Popularity.sort_values(ascending=False)
    popular_item = sorted_Popularity.index.tolist()
    user_data = data.groupby('userId')
    for user, value in user_data:
        interacted_item = value['itemId'].tolist()
        index = 0
        for j in range(numOfRecommend):
            while popular_item[index] in interacted_item:
                index = index + 1
            rec.append({'userId': str(user), 'itemId': str(popular_item[index]), 'member_status': '0'})
            index = index + 1

    rec = pd.DataFrame(rec)
    rec.columns = ['userId','itemId','rating']
    rec.to_csv(output_dir + filename[:16] + '_recommendation.csv', index = False)



if __name__ == '__main__':
    root_path = "../data/processed_ml-1m/"
    output_dir = "../data/processed_ml-1m/processed_Nml1m"
    datalist = ["target_nonmember","shadow_nonmember"]
    for filename in datalist:
        recommend(root_path,filename,output_dir)
'''