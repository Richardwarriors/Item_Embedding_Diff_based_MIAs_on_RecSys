import torch
import pandas as pd
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import save_checkpoint, use_optimizer
from metrics import MetronAtK


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()
        self.best_hr = 0
        self.best_model = None
        self.best_epoch = 0

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            #print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            negative_users, negative_items = evaluate_data[2], evaluate_data[3]
            if self.config['use_cuda'] is True:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                negative_users = negative_users.cuda()
                negative_items = negative_items.cuda()

        if self.config['use_bachify_eval'] == False:    
            test_scores = self.model(test_users, test_items)
            negative_scores = self.model(negative_users, negative_items)
        else:
            test_scores = []
            negative_scores = []
            bs = self.config['batch_size']
            for start_idx in range(0, len(test_users), bs):
                end_idx = min(start_idx + bs, len(test_users))
                batch_test_users = test_users[start_idx:end_idx]
                batch_test_items = test_items[start_idx:end_idx]
                test_scores.append(self.model(batch_test_users, batch_test_items))
            for start_idx in tqdm(range(0, len(negative_users), bs)):
                end_idx = min(start_idx + bs, len(negative_users))
                batch_negative_users = negative_users[start_idx:end_idx]
                batch_negative_items = negative_items[start_idx:end_idx]
                negative_scores.append(self.model(batch_negative_users, batch_negative_items))
            test_scores = torch.concatenate(test_scores, dim=0)
            negative_scores = torch.concatenate(negative_scores, dim=0)


            if self.config['use_cuda'] is True:
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                test_scores = test_scores.cpu()
                negative_users = negative_users.cpu()
                negative_items = negative_items.cpu()
                negative_scores = negative_scores.cpu()
            self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
        self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
        
        print('[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id, hit_ratio, ndcg))
        if hit_ratio > self.best_hr:
            self.best_hr = hit_ratio
            self.best_epoch = epoch_id
            #self.best_model = self.model.state_dict()
            self.best_model = self.model

        #self.generate_recommendations(top_k = 100)
        print('[Best Epoch {}] HR = {:.4f}'.format(self.best_epoch, self.best_hr))
        return hit_ratio, ndcg

    def generate_recommendations(self, sample_generator, top_k=100):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.best_model.eval()  # 确保模型处于评估模式

        # 加载user_mapping和item_mapping
        user_mapping_df = pd.read_csv('../data/processed_ml-1m/processed_Nml1m/userid_shadow.csv')
        item_mapping_df = pd.read_csv('../data/processed_ml-1m/processed_Nml1m/itemid_shadow.csv')

        # 转换为字典 {new_userId: original_userId} 和 {new_itemId: original_itemId}
        user_mapping = dict(zip(user_mapping_df['inner_uid'], user_mapping_df['raw_uid']))
        #print(user_mapping)
        item_mapping = dict(zip(item_mapping_df['inner_iid'], item_mapping_df['raw_iid']))

        user_nointeracted_item = sample_generator.negatives
        recommendations = []  # 用来存储每个用户的推荐结果

        with torch.no_grad():
            all_users = torch.arange(self.config['num_users'])

            if self.config['use_cuda']:
                all_users = all_users.cuda()

            for user in tqdm(all_users):
                negative_items = user_nointeracted_item.loc[user_nointeracted_item['userId'] == user.item(), 'negative_items'].values[0]
                candidate_items = torch.LongTensor(list(negative_items))

                if self.config['use_cuda']:
                    candidate_items = candidate_items.cuda()

                # Predict scores for all candidate items
                user_tensor = user.repeat(len(candidate_items))
                item_scores = self.best_model(user_tensor, candidate_items)
                top_items = torch.topk(item_scores.view(-1), k=top_k).indices

                # Map new IDs to original IDs
                recommended_items = [candidate_items[i].item() for i in top_items]

                original_user = user_mapping[user.item()]
                original_items = [item_mapping[item] for item in recommended_items]

                # Append recommendations for the user
                recommendations.append([original_user] + original_items)

        # Convert recommendations to DataFrame
        columns = ['raw_uid'] + [f'rec_iid{i + 1}' for i in range(top_k)]
        rec_df = pd.DataFrame(recommendations, columns=columns)

        # Save recommendations to CSV
        rec_df.to_csv('../data/processed_ml-1m/processed_Nml1m/shadow_recommendations.csv', index=False)
        print("Recommendations saved to 'shadow_recommendations.csv'")

        return rec_df

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)
