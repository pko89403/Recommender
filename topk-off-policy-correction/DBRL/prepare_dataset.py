# %%
import os
from collections import defaultdict
import math
import random

import torch
import numpy as np
import pandas as pd

# %%
resource_path = "topk-off-policy-correction/DBRL/resources"
user_emb_file = "tianchi_user_embeddings.npy"
item_emb_file = "tianchi_item_embeddings.npy"
behavior_file = "tianchi.csv"

# %%
user_embeddings = None
item_embeddings = None

with open(os.path.join(resource_path, user_emb_file), "rb") as f:
    user_embeddings = np.load(f)
    
with open(os.path.join(resource_path, item_emb_file), "rb") as f:
    item_embeddings = np.load(f)

user_embeddings.shape, item_embeddings.shape

# %%
N_EPOCHS = 100 
HIST_NUM = 10 # num of history items to consider
N_REC = 10 # num of items to recommend
BATCH_SIZE = 128
HIDDEN_SIZEZ = 64
LR = 1e-5
WEIGHT_DECAY = 0.
GAMMA = 0.99
SESS_MODE = "interval" # specify when to end a session
SEED = 0

# %%
from traitlets import default


def process_data(
    path,
    columns=None,
    test_size=0.2,
    time_col="time",
    sess_mode="one",
    interval=None,
    reward_shape=None,
    shuffle=False,
    pad_unknown=True,
    filter_unknown=False,
):
    column_names = columns if columns is not None else None
    data = pd.read_csv(path, sep=",", names=column_names)
    assert time_col in data.columns, "must specify correct time column name..."
    
    # split_by_ratio
    n_users = data.user.nunique()
    user_indices = data.user.to_numpy()
    
        # groupby_user
    users, user_position, user_counts = np.unique(
        user_indices,
        return_inverse=True, # 각 값이 몇번째 고유 원소에 매칭 되는지에 대한 inverse 정보
        return_counts=True, # 각 고유 원소가 등장한 총 횟수를 나타내는 counts 정보
    )
    user_split_indices = np.split(
        np.argsort(user_position, kind="mergesort"),
        np.cumsum(user_counts)[:-1],
    )

    split_indices_all = [[], []]
    for u in range(n_users):
        u_data = user_split_indices[u]
        u_data_len = len(u_data)
        if u_data_len <= 3:
            # 짧은 세션은 그냥 가져감
            split_indices_all[0].extend(u_data)
        else:
            # 한 유저의 세션 중에서 test_size 만큼의 비중의 뒷부분을 test로 사용함.
            train_threshold = round((1 - test_size) * u_data_len)
            split_indices_all[0].extend(list(u_data[:train_threshold]))
            split_indices_all[1].extend(list(u_data[train_threshold:]))
    
    if shuffle:
        split_data_all = tuple(
            np.random.permutation(data[idx]) for idx in split_indices_all
        )
    else:
        split_data_all = list(data.iloc[idx] for idx in split_indices_all)
    
    if pad_unknown:
        # _pad_unknown_item
        train_data, test_data = split_data_all
        train_n_items = train_data.item.nunique()
        train_unique_items = set(train_data.item.tolist())
        test_data.loc[~test_data.item.isin(train_unique_items), "item"] = train_n_items
        split_data_all = (train_data, test_data)
    elif filter_unknown:
        # _filter_unknown_user_item
        train_data, test_data = split_data_all
        train_unique_values = dict(user=set(train_data.user.tolist()),
                                   item=set(train_data.item.tolist()))

        print(f"test data size before filtering: {len(test_data)}")
        out_of_bounds_row_indices = set()
        for col in ["user", "item"]:
            for j, val in enumerate(test_data[col]):
                if val not in train_unique_values[col]:
                    out_of_bounds_row_indices.add(j)
        
        mask = np.arange(len(test_data))
        test_data_clean = test_data[~np.isin(mask, list(out_of_bounds_row_indices))]
        print(f"test data size after filtering: {len(test_data_clean)}")
        split_data_all = (train_data, test_data)

    # map unique value
    train_data, test_data = split_data_all
    """
         user    item        label  time  sex  age  pur_power  category     shop   
    10   970632  23789102    pv     1547    1    1          6     12049   767338   
    11   970632   7581033    pv     2165    1    1          6     12049  2201114   
    12   970632  28318368    pv     2441    1    1          6     12049  3934899   
    13   970632  15438137    pv     2498    1    1          6     12049   715961   
    14   970632  19771111    pv     2533    1    1          6     12049   399917   
    15   970632   3879903    pv     2598    1    1          6     14393  3246567   
    17   970632   2006014    pv     2865    1    1          6     12049    13310   

    """
    # userID를 index로 매핑 ex) userID(1269894) -> userIndex(0)
    # itenId를 index로 매핑
    for col in ["user", "item"]:
        counts = train_data[col].value_counts()
        freq = counts.index.tolist()
        mapping = dict(zip(freq, range(len(freq))))
        train_data[col] = train_data[col].map(mapping)
        test_data[col] = test_data[col].map(mapping)
        if test_data[col].isnull().any():
            col_type = train_data[col].dtype
            test_data[col].fillna(len(freq), inplace=True)
            test_data[col] = test_data[col].astype(col_type)
    
    n_users = train_data.user.nunique()
    n_items = train_data.item.nunique()
    
    # train_user_consumed = build_interaction(train_data)
    train_user_consumed = defaultdict(list)
    for u, i in zip(train_data.user.tolist(), train_data.item.tolist()):
        train_user_consumed[u].append(i)
    test_user_consumed = defaultdict(list)
    for u, i in zip(test_data.user.tolist(), test_data.item.tolist()):
        test_user_consumed[i].append(u)

    # build_reward
    train_rewards = test_rewards = None
    if reward_shape is not None:
        train_label_all = defaultdict(list)
        for u, l in zip(train_data.user.tolist(), train_data.label.tolist()):
            train_label_all[u].append(l)
        
        train_reward_all = defaultdict(dict)
        for user, label in train_label_all.items():
            for key, rew in reward_shape.items():
                index = np.where(np.array(label) == key)[0]
                if len(index) > 0:
                    train_reward_all[user].update({key: index})
        train_rewards = train_reward_all

        test_label_all = defaultdict(list)
        for u, l in zip(test_data.user.tolist(), test_data.label.tolist()):
            test_label_all[u].append(l)
        
        test_reward_all = defaultdict(dict)
        for user, label in test_label_all.items():
            for key, rew in reward_shape.items():
                index = np.where(np.array(label) == key)[0]
                if len(index) > 0:
                    test_reward_all[user].update({key: index})
        test_rewards = test_reward_all
    else:
        train_rewards = test_rewards = None
        
    print(train_rewards)
    
    # train_sess_end = build_sess_end(train_data, sess_mode, time_col, interval)
    if sess_mode == "one":
        # 유저가 본 전체를 한 세션으로 볼 것인가?
        train_sess_end = train_data.groupby("user").apply(len).to_dict()
        test_sess_end = test_data.groupby("user").apply(len).to_dict()
    elif sess_mode == "interval":
        train_sess_times = train_data[time_col].astype('int').to_numpy()
        # train_user_split_indices = groupby_user(train_data.user.to_numpy())
        train_users, train_user_position, train_user_counts = np.unique(
            train_data.user.to_numpy(),
            return_inverse=True,
            return_counts=True
        )
        train_user_split_indices = np.split(
            np.argsort(train_user_position, kind="mergesort"),
            np.cumsum(train_user_counts)[:-1]
        )
        train_sess_end = dict()
        for u in range(len(train_user_split_indices)):
            train_u_idxs = train_user_split_indices[u]
            train_user_ts = train_sess_times[train_u_idxs]
            # if neighbor time interval > sess_interval, then end of a session
            # 유저가 아이템을 소비한 시간 사이의 간격을 구해서 interval보다 크면 세션의 끝으로 표기
            # np.where(np.ediff1d(np.array([1207, 1231, 1334, 1699, 18880])) > 100)[0] -> array([1, 2, 3])
            train_sess_end[u] = np.where(np.ediff1d(train_user_ts) > interval)[0]
            
        test_sess_times = test_data[time_col].astype('int').to_numpy()
        test_users, test_user_position, test_user_counts = np.unique(
            test_data.user.to_numpy(),
            return_inverse=True,
            return_counts=True,
        )
        test_user_split_indices = np.split(
            np.argsort(test_user_position, kind="mergesort"),
            np.cumsum(test_user_counts)[:-1]
        )
        test_sess_end = dict()
        for u in range(len(test_user_split_indices)):
            test_u_idxs = test_user_split_indices[u]
            test_user_ts = test_sess_times[test_u_idxs]
            # if neighbor time interval > sess_interval, then end of a session
            test_sess_end[u] = np.where(np.ediff1d(test_user_ts) > interval)[0]
    else:
        raise ValueError("sess_mode must be 'one' or 'interval'")
    
    result = (
        n_users,
        n_items,
        train_user_consumed,
        test_user_consumed,
        train_sess_end,
        test_sess_end,
        train_rewards,
        test_rewards
    )

    return result


(
    n_users,
    n_items,
    train_user_consumed,
    test_user_consumed,
    train_sess_end,
    test_sess_end,
    train_rewards,
    test_rewards,
) = process_data(
    path=os.path.join(resource_path, behavior_file),
    columns=["user", "item", "label", "time", "sex", "age", "pur_power", "category", "shop", "brand"],
    test_size=0.2,
    time_col="time",
    sess_mode=SESS_MODE,
    interval=int(60*60),
    reward_shape={"pv": 1., "cart": 2., "fav": 2., "buy": 3.},
    shuffle=False,
)


# %%
n_users, n_items

# %%
train_user_consumed, test_user_consumed

# %%
train_sess_end, test_sess_end

# %%
train_rewards, test_rewards

# %%
from torch.utils.data import Dataset, DataLoader

class RLDataset(Dataset):
    def __init__(self, data, has_return=False):
        self.data = data
        self.has_return = has_return
    
    def __getitem__(self, index):
        user = self.data["user"][index]
        items = self.data["item"][index]
        if not self.has_return:
            res = {
                "user": user,
                "item": items[:-1],
                "action": items[-1],
                "reward": self.data["reward"][index],
                "done": self.data["done"][index],
                "next_item": items[1:]
            }
        else:
            if "beta_label" in self.data:
                res = {
                    "user": user,
                    "item": items[:-1],
                    "action": items[-1],
                    "return": self.data["return"][index],
                    "beta_user": self.data["beta_user"][index],
                    "beta_item": self.data["beta_item"][index],
                    "beta_label": self.data["beta_label"][index]
                }
            else:
                res = {
                    "user": user,
                    "item": items[:-1],
                    "action": items[-1],
                    "return": self.data["return"][index]
                }
        return res
    
    def __len__(self):
        return len(self.data["item"])

def build_dataloader(
    n_users,
    n_items,
    hist_num,
    train_user_consumed,
    test_user_consumed,
    batch_size,
    sess_mode="one",
    train_sess_end=None,
    test_sess_end=None,
    n_workers=0,
    compute_return=False,
    neg_sample=None,
    train_rewards=None,
    test_rewards=None,
    reward_shape=None,
):
    TRAIN = True
    NEG_SAMPLE= None

    if not compute_return:
        # train_session = build_session()
        
        user_sess, item_sess, reward_sess, done_sess = [], [], [], []
        user_consumed_set = {
            u: set(items) for u, items in train_user_consumed.items()
        }
        for u in range(n_users):
            if TRAIN:
                items = np.asarray(train_user_consumed[u])
            else:
                items = np.asarray(
                    train_user_consumed[u][-hist_num:] + test_user_consumed[u]
                )
            
            hist_len = len(items)
            # expanded_items = pad_session(hist_len, hist_num, items, pad_val=n_items)
            """Pad items sequentially.

            For example, a user's whole item interaction is [1,2,3,4,5],
            then it will be converted to the following matrix:
            x x x x 1 2
            x x x 1 2 3
            x x 1 2 3 4
            x 1 2 3 4 5

            Where x denotes the padding-value. Then for the first line, [x x x x 1]
            will be used as state, and [2] as action.

            If the length of interaction is longer than hist_num, the rest will be
            handled by function `rolling_window`, which converts the rest of
            interaction to:
            1 2 3 4 5 6
            2 3 4 5 6 7
            3 4 5 6 7 8
            ...

            In this case no padding value is needed. So basically every user in the
            data will call `pad_session`, but only users with long interactions will
            need to call `rolling_window`.
            """
            sess_len = hist_len - 1 if hist_len - 1 < hist_num - 1 else hist_num - 1
            # n_items는 패딩 인덱스 값
            session_first = np.full((sess_len, hist_num + 1), n_items, dtype=np.int64)
            for i in range(sess_len):
                offset = i + 2
                session_first[i, -offset:] = items[:offset]
            expanded_items = session_first
            
            if hist_len > hist_num:
                # full_size_sess = rolling_window(items, hist_num + 1)
                assert (hist_num + 1) <= items.shape[-1], "window size too large..."
                shape = items.shape[:-1] + (items.shape[-1] - (hist_num+1) + 1, (hist_num+1))
                strides = items.strides + (items.strides[-1],)
                full_size_sess = np.lib.stride_tricks.as_strided(items, shape=shape, strides=strides)
                expanded_items = np.concatenate(
                    [expanded_items, full_size_sess],
                    axis=0
                )

            if TRAIN and NEG_SAMPLE is not None:
                # expanded_items, num_neg, _ = sample_neg_session(
                #     expanded_items, user_consumed_set[u], n_items, NEG_SAMPLE
                # )
                size = len(expanded_items)
                if size <= 3:
                    expanded_items = expanded_items
                    num_neg = 0
                
                num_neg = size // 2
                item_sampled = []
                for _ in range(num_neg):
                    item_neg = math.floor(n_items * random.random())
                    while item_neg in user_consumed_set[u]:
                        item_neg = math.floor(n_items * random.random())
                    item_sampled.append(item_neg)
                
                if NEG_SAMPLE == "random":
                    indices = np.random.choice(size, num_neg, replace=False)
                else:
                    indices = np.arange(size - num_neg, size)
                assert len(indices) == num_neg, "indices and num_neg must equal."
                neg_items = expanded_items[indices]
                neg_items[:, -1] = item_sampled
                expanded_items = np.concatenate([expanded_items, neg_items], axis=0)
                num_neg = num_neg
            
            sess_len = len(expanded_items)
            user_sess.append(np.tile(u, sess_len))
            item_sess.append(expanded_items)
            
            if reward_shape is not None:
                # reward = assign_reward(
                #     sess_len, u, train, train_rewards, test_rewards,
                #     train_user_consumed, hist_num, reward_shape
                # )
                reward = np.ones(sess_len, dtype=np.float32)
                
                if TRAIN and train_rewards is not None:
                    for label, index in train_rewards[u].items():
                        # skip first item as it will never become label
                        index = index -1
                        index = index[index >= 0]
                        if len(index) > 0:
                            reward[index] = reward_shape[label]
                elif (
                        not TRAIN
                        and test_rewards is not None
                        and train_rewards is not None
                ):
                    train_len = len(train_user_consumed[u])
                    train_dummy_reward = np.ones(train_len, dtype=np.float32)
                    boundary = (
                        hist_num - 1
                        if hist_num - 1 < train_len - 1
                        else train_len - 1
                    )
                    for label, index in train_rewards[u].items():
                        index = index - 1
                        index = index[index >= 0]
                        if len(index) > 0:
                            train_dummy_reward[index] = reward_shape[label]
                    reward[:boundary] = train_dummy_reward[-boundary:]
                    
                    if test_rewards[u]:
                        for label, index in test_rewards[u].items():
                            index = index + boundary
                            reward[index] = reward_shape[label]   
            else:
                reward = np.ones(sess_len, dtype=np.float32)
            
            if TRAIN and NEG_SAMPLE is not None and num_neg > 0:
                reward[-num_neg:] = 0.
            reward_sess.append(reward)
            
            done = np.zeros(sess_len, dtype=np.float32)
            if TRAIN and SESS_MODE == "interval":
                end_mask = train_sess_end[u]
                done[end_mask] = 1.
            if TRAIN and NEG_SAMPLE is not None and num_neg > 0:
                done[-num_neg - 1] = 1.
            else:
                done[-1] = 1.
            done_sess.append(done)
            
        res = {
            "user": np.concatenate(user_sess),
            "item": np.concatenate(item_sess, axis=0),
            "reward": np.concatenate(reward_sess),
            "done": np.concatenate(done_sess)
        }
        
        train_session = res
    else:
        GAMMA = 0.99
        NORMALIZE = False
        
        (
            user_sess,
            item_sess,
            return_sess,
            beta_users,
            beta_items,
            beta_labels
        ) = [], [], [], [], [], []
        user_consumed_set = {
            u: set(items) for u, items in train_user_consumed.items()
        }
        for u in range(n_users):
            if u == 1:
                break
            if TRAIN:
                items = np.asarray(train_user_consumed[u])
            else:
                items = np.asarray(
                    train_user_consumed[u][-hist_num:] + test_user_consumed[u]
                )
            
            hist_len = len(items)
            # expanded_items = pad_session(hist_len, hist_num, items, pad_val=n_items)
            sess_len = hist_len - 1 if hist_len -1 < hist_num -1 else hist_num - 1
            session_first = np.full((sess_len, hist_num + 1), n_items, dtype=np.int64)
            for i in range(sess_len):
                offset = i + 2
                session_first[i, -offset:] = items[:offset]
            expanded_items = session_first
            
            
            if hist_len > hist_num:
                # full_size_sess = rolling_window(items, hist_num + 1)
                window = hist_num + 1
                assert window <= items.shape[-1], "window size too long..."
                shape = items.shape[:-1] + (items.shape[-1] - window + 1, window)
                strides = items.strides + (items.strides[-1],)
                full_size_sess = np.lib.stride_tricks.as_strided(items, shape=shape, strides=strides)
                expanded_items = np.concatenate(
                    [expanded_items, full_size_sess],
                    axis=0
                )

            if TRAIN and NEG_SAMPLE is not None:
                # neg_items, num_neg, expanded_items = sample_neg_session(
                #     expanded_items, user_consumed_set[u], n_items, NEG_SAMPLE
                # )
                size = len(expanded_items)
                if size <= 3:
                    neg_items = expanded_items
                    num_neg = 0
                    expanded_items=expanded_items
                else:
                    num_neg = size // 2
                    item_sampled = []
                    
                    for _ in range(num_neg):
                        item_neg = math.floor(n_items * random.random())
                        while item_neg in user_consumed_set[u]:
                            item_neg = math.floor(n_items * random.random())
                        item_sampled.append(item_neg)
                    
                    if NEG_SAMPLE == "random":
                        indices = np.random.choice(size, num_neg, replace=False)
                    else:
                        indices = np.arange(size - num_neg, size)
                    assert len(indices) == num_neg, "indices and num_neg must equal."
                    neg_items = expanded_items[indices]
                    neg_items[:, -1] = item_sampled
                    
                    neg_items = np.concatenate([expanded_items, neg_items], axis=0)
                    num_neg = num_neg
                    expanded_items = expanded_items
            
            sess_len = len(expanded_items)
            user_sess.append(np.tile(u, sess_len))
            item_sess.append(expanded_items)

            if reward_shape is not None:
                # reward = assign_reward(
                #     sess_len, u, TRAIN, train_rewards, test_rewards,
                #     train_user_consumed, hist_num, reward_shape
                # )
                reward = np.ones(sess_len, dtype=np.float32)
                if TRAIN and train_rewards is not None:
                    for label, index in train_rewards[u].items():
                        # skip first item as it will never become label
                        index = index - 1
                        index = index[index >= 0]
                        if len(index) > 0:
                            reward[index] = reward_shape[label]
                elif (
                    not TRAIN
                    and test_rewards is not None
                    and train_rewards is not None
                ):
                    train_len = len(train_user_consumed[u])
                    train_dummy_reward = np.ones(train_len, dtype=np.float32)
                    boundary = (
                        hist_num - 1
                        if hist_num - 1 < train_len - 1
                        else train_len - 1
                    )
                    for label, index in train_rewards[u].items():
                        index = index - 1
                        index = index[index >= 0]
                        if len(index) > 0:
                            train_dummy_reward[index] = reward_shape[label]
                    reward[:boundary] = train_dummy_reward[-boundary:]
                
                    if test_rewards[u]:
                        for label, index in test_rewards[u].items():
                            index = index + boundary
                            reward[index] = reward_shape[label]
                    
            else:
                reward = np.ones(sess_len, dtype=np.float32)

            sess_end_u = (
                train_sess_end[u] + 1
                if TRAIN and SESS_MODE == "interval"
                else None
            )
            
            # return_sess.append(
            #     compute_returns(reward, gamma, sess_end_u, normalize=False)
            # )
            total_returns = []
            if sess_end_u is None:
                last_val = 0
                for r in reversed(reward):
                    last_val = r + GAMMA * last_val
                    total_returns.append(last_val)
                total_returns.reverse()
            else:
                for rew in np.split(reward, sess_end_u):
                    returns = []
                    last_val = 0
                    for r in reversed(rew):
                        last_val = r + GAMMA * last_val
                        returns.append(last_val)
                    returns.reverse()
                    total_returns.extend(returns)
            
            total_returns = np.asarray(total_returns)
            if NORMALIZE:
                total_returns /= (np.linalg.norm(total_returns) +  1e-7)

            return_sess.append(total_returns)
            
            if TRAIN and NEG_SAMPLE is not None and num_neg > 0:
                beta_len = len(neg_items)
                beta_users.append(np.tile(u, beta_len))
                beta_items.append(neg_items[:, :-1])
                beta_labels.append(neg_items[:, -1])

        if TRAIN and NEG_SAMPLE is not None and num_neg > 0:
            res = {"user": np.concatenate(user_sess),
                   "item": np.concatenate(item_sess, axis=0),
                   "return": np.concatenate(return_sess),
                   "beta_user": np.concatenate(beta_users),
                   "beta_item": np.concatenate(beta_items, axis=0),
                   "beta_label": np.concatenate(beta_labels)}
        else:
            res = {
                    "user": np.concatenate(user_sess),
                    "item": np.concatenate(item_sess, axis=0),
                    "return": np.concatenate(return_sess),
            }

            np.save("train_session.npy", res)
        train_session = res

    if not compute_return:
        train_rl_data = RLDataset(train_session)
    else:
        train_rl_data = RLDataset(train_session, has_return=True)
        
    train_rl_loader = DataLoader(
        train_rl_data,
        batch_size=1,
        shuffle=True,
        num_workers=n_workers
    )
    
    for i in train_rl_loader:
        print(i)
        break
    
    
    return train_rl_loader
        

reward_map = {"pv": 1., "cart": 2., "fav": 2., "buy": 3.}
train_rl_loader = build_dataloader(
    n_users=n_users,
    n_items=n_items,
    hist_num=10,
    train_user_consumed=train_user_consumed,
    test_user_consumed=test_user_consumed,
    batch_size=10,
    sess_mode="interval",
    train_sess_end=train_sess_end,
    test_sess_end=test_sess_end,
    n_workers=0,
    compute_return=True,
    neg_sample=False,
    train_rewards=train_rewards,
    test_rewards=test_rewards,
    reward_shape=reward_map
)