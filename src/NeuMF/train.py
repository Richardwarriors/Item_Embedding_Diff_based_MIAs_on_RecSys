import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator

gmf_config = {'alias': 'gmf_ml1ms',
              'num_epoch': 100,
              'batch_size': 256,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 1015,
              'num_items': 3406,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0,  # 0.01
              'weight_init_gaussian': True,
              'use_cuda': True,
              'use_bachify_eval': True,
              'device_id': 1,
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp_ml1ms',
              'num_epoch': 100,
              'batch_size': 256,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 1015,
              'num_items': 3406,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0001,  # MLP model is sensitive to hyper params
              'weight_init_gaussian': True,
              'use_cuda': True,
              'use_bachify_eval': True,
              'device_id': 1,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_ml1ms_Epoch29_HR0.6443_NDCG0.3507.model'),
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config ={'alias': 'neumf_ml1ms',
                'num_epoch': 100,
                'batch_size': 256,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 1015,
                'num_items': 3406,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.001,
                'weight_init_gaussian': True,
                'use_cuda': True,
                'use_bachify_eval': True,
                'device_id': 1,
                'pretrain': True,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_ml1ms_Epoch29_HR0.6443_NDCG0.3507.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_ml1ms_Epoch57_HR0.6374_NDCG0.3468.model'),
                'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }

# Load Data
data_dir = '../data/processed_ml-1m/shadow_member.csv'
data_rating = pd.read_csv(data_dir, sep=',')
data_rating.columns = ['uid', 'mid', 'rating', 'timestamp']

#data_rating['uid'] = pd.to_numeric(data_rating['uid'], errors='coerce').astype('Int64')
#data_rating['mid'] = pd.to_numeric(data_rating['mid'], errors='coerce').astype('Int64')
#data_rating['rating'] = pd.to_numeric(data_rating['rating'], errors='coerce').astype('Int64')
#data_rating['timestamp'] = pd.to_numeric(data_rating['timestamp'], errors='coerce').astype('Int64')

user_id = data_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
data_rating = pd.merge(data_rating, user_id, on=['uid'], how='left')
item_id = data_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
data_rating = pd.merge(data_rating, item_id, on=['mid'], how='left')
data_rating = data_rating[['userId', 'itemId', 'rating', 'timestamp']]
print('Range of userId is [{}, {}]'.format(data_rating.userId.min(), data_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(data_rating.itemId.min(), data_rating.itemId.max()))

# 检测 NaN 或 inf
has_nan_or_inf = data_rating.apply(lambda x: x.isna().any() or np.isinf(x).any())
print("是否有 NaN 或 inf 值:\n", has_nan_or_inf)

# 统计每列 NaN 或 inf 的数量
nan_or_inf_count = data_rating.apply(lambda x: x.isna().sum() + np.isinf(x).sum())
print("\n每列 NaN 或 inf 的数量:\n", nan_or_inf_count)

# 将新的用户ID映射到原始的用户ID，新ID作为key，原ID作为value
user_mapping = dict(zip(user_id['userId'], user_id['uid']))
# 将映射保存为 DataFrame 并导出为 CSV
user_mapping_df = pd.DataFrame(list(user_mapping.items()), columns=['inner_uid', 'raw_uid'])
user_mapping_df.to_csv('../data/processed_ml-1m/processed_Nml1m/userid_shadow.csv', index=False)

item_mapping = dict(zip(item_id['itemId'], item_id['mid']))
item_mapping_df = pd.DataFrame(list(item_mapping.items()), columns=['inner_iid', 'raw_iid'])
item_mapping_df.to_csv('../data/processed_ml-1m/processed_Nml1m/itemid_shadow.csv', index=False)


# DataLoader for training
sample_generator = SampleGenerator(ratings=data_rating)
evaluate_data = sample_generator.evaluate_data

# Specify the exact model
#config = gmf_config
#engine = GMFEngine(config)
#config = mlp_config
#engine = MLPEngine(config)
config = neumf_config
engine = NeuMFEngine(config)
for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    engine.save(config['alias'], epoch, hit_ratio, ndcg)

#create user-recommended dataframe
recommendations = engine.generate_recommendations(sample_generator, top_k=100)


