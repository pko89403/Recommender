import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from models import Beta, ChooseREINFORCE, DiscreteActor, Critic
from env import FrameEnv, DataPath

dataPath = DataPath(
    base="topk-off-policy-correction/data/ml-1m/",
    ratings="ratings.csv",
    embeddings="ml20_pca128.pkl",
    use_cache=False,
)

path_embedding = "topk-off-policy-correction/data/ml-1m/ml20_pca128.pkl"

movie_embeddings_key_dict = pickle.load(open(path_embedding, "rb"))

keys = list(sorted(movie_embeddings_key_dict.keys()))
key_to_id = dict(zip(keys, range(len(keys))))
id_to_key = dict(zip(range(len(keys)), keys))

items_embeddings_id_dict = {}
for k in movie_embeddings_key_dict.keys():
    items_embeddings_id_dict[key_to_id[k]] = movie_embeddings_key_dict[k]

items_embeddings_id_dict = torch.stack(
    [items_embeddings_id_dict[i] for i in range(len(items_embeddings_id_dict))]
)
embeddings = items_embeddings_id_dict
key_to_id = key_to_id
id_to_key = id_to_key
num_items = embeddings.shape[0]

path_ratings = "topk-off-policy-correction/data/ml-1m/ratings.csv"
ratings = pd.read_csv(
    path_ratings,
    names=[
        "userId",
        "movieId",
        "rating",
        "timestamp",
    ]
)

ratings["rating"] = ratings["rating"].apply(lambda i: 2* (i - 2.5))
ratings["movieId"] = ratings["movieId"].apply(key_to_id.get).fillna(0.0)

users = ratings[["userId", "movieId"]].groupby(["userId"]).size()
frames_size = 10
users = users[users > frames_size].sort_values(ascending=False).index

ratings_grp = (
    ratings.sort_values(by="timestamp")
    .set_index("userId")
    .drop("timestamp", axis=1)
    .groupby("userId")
)

user_dict = {}

def app(x):
    userid = x.index[0]
    user_dict[userid] = {}
    user_dict[userid]['items'] = x["movieId"].values
    user_dict[userid]['ratings'] = x["rating"].values

ratings_grp.apply(app)


test_size = 0.05

train_users, test_users = train_test_split(users, test_size=test_size)
train_users = pd.Series(dict([(i, user_dict[i]["items"].shape[0]) for i in users])).sort_values(ascending=False).index

class UserDataset(Dataset):
    def __init__(self, users, user_dict):
        self.users = users
        self.user_dict = user_dict
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        idx = self.users[idx]
        group = self.user_dict[idx]
        items = group["items"][:]
        rates = group["ratings"][:]
        size = items.shape[0]
        return {"items": items, "rates": rates, "sizes": size, "users": idx}

train_user_dataset = UserDataset(
    train_users,
    user_dict,
)

def rolling_window(a, window):
    """numpy를 사용한 시퀀스 frame_size 만큼 window
    
          [2567. 1240.  419. ...  490. 1728. 1498.]
          -> [[2567. 1240.  419. ... 3710. 3741. 3703.]
              [1240.  419. 2543. ... 3741. 3703. 3720.]
              ...
              [2725. 3823. 3791. ... 1380.  490. 1728.]
              [3823. 3791. 3289. ...  490. 1728. 1498.]]

    Args:
        a (_type_): _description_
        window (_type_): _description_

    Returns:
        _type_: _description_
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    res = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return res

def prepare_batch_wrapper(batch):
    items_t, ratings_t, sizes_t, users_t = [], [], [], []

    for i in range(len(batch)):
        items_t.append(batch[i]["items"])
        ratings_t.append(batch[i]["rates"])
        sizes_t.append(batch[i]["sizes"])
        users_t.append(batch[i]["users"])

    frame_size = 10
    
    items_t = np.concatenate([rolling_window(i, frame_size + 1) for i in items_t], 0)
    ratings_t = np.concatenate([rolling_window(i, frame_size + 1) for i in ratings_t], 0)
    
    items_t = torch.tensor(items_t)
    users_t = torch.tensor(users_t)
    ratings_t = torch.tensor(ratings_t).float()
    sizes_t = torch.tensor(sizes_t)
    
    # frame_size is 10
    items_emb = embeddings[items_t.long()] # shape: (b, 11, 128)
    b_size = ratings_t.size(0)
    
    
    items = items_emb[:, :-1, :].view(b_size, -1) # start from 0 (b, 10, 128) -> (b, 10*1280)
    next_items = items_emb[:, 1:, :].view(b_size, -1) # start from 1 (b, 10, 128) -> (b, 10*1280)
    ratings = ratings_t[:, :-1] # (b, 10)
    next_ratings = ratings_t[:, 1:]

    state = torch.cat([items, ratings], 1) # (b, 10*1280 + 10)
    next_state = torch.cat([next_items, next_ratings], 1)
    action = items_t[:, -1] # last item's emb
    reward = ratings[:, -1] # last item's rating
    
    done = torch.zeros(b_size) # ?
    done[torch.cumsum(sizes_t-frame_size, dim=0) - 1] = 1 # done은 어디에 쓰지?

    one_hot_action = torch.zeros(b_size, num_items)
    one_hot_action.scatter_(1, action.view(-1, 1).long(), 1)

    batch = {
        "state": state, # items + ratings
        "action": one_hot_action, # last item
        "reward": reward, # last item rating
        "next_state": next_state, # next items + next ratings
        "done": done, # each user history end
        "meta" : {"users": users_t, "sizes": sizes_t}, # userID, hist_length
    }
    
    return batch

train_dataloader = DataLoader(
    train_user_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=prepare_batch_wrapper
)

rl_params = {
    "reinforce":    ChooseREINFORCE(ChooseREINFORCE.basic_reinforce),
    "reinforce_corr": ChooseREINFORCE(ChooseREINFORCE.reinforce_with_correction),
    "reinforce_topk_corr": ChooseREINFORCE(ChooseREINFORCE.reinforce_with_TopK_correction),
    "gamma":        0.99,
    "min_value":    -10,
    "max_value":    10,
    "policy_step":  10,
    "soft_tau":     0.001,
    "policy_lr":    1e-5,
    "value_lr":     1e-5,
    "actor_weight_init":    54e-2,
    "critic_weight_init":   6e-1,
    "K": 10, # For Correction
    
}

num_inputs = 1290
hidden_dim = 2048
num_items = embeddings.shape[0]

networks = {
    "value_net":            Critic(num_inputs, num_items, hidden_dim, rl_params["critic_weight_init"]),
    "target_value_net":     Critic(num_inputs, num_items, hidden_dim, rl_params["actor_weight_init"]),
    "policy_net":           DiscreteActor(hidden_dim, num_inputs, num_items),
    "target_policy_net":    DiscreteActor(hidden_dim, num_inputs, num_items).eval(),
    "beta_net":             Beta(num_inputs, num_items),
}

optimizer = {
    "value_optimizer": torch.optim.AdamW(networks["value_net"].parameters(),
                                        lr=rl_params["value_lr"], weight_decay=1e-2),
    "policy_optimizer": torch.optim.AdamW(networks["policy_net"].parameters(),
                                         lr=rl_params["policy_lr"], weight_decay=1e-5)
}

loss = {
    "tst": {"value": [], "policy": [], "step": []},
    "train": {"value": [], "policy": [], "step": []}
}
debug = {}

learn = True

for step, batch in enumerate(train_dataloader):
    state = batch["state"]
    action = batch["action"]
    reward = batch["reward"].unsqueeze(1)
    next_state = batch["next_state"]
    done = batch["done"].unsqueeze(1)

    # Simple ACTOR_CRITIC
    # predicted_action, predicted_probs = networks["policy_net"].select_action(state)
    # Off-Policy-Correction
    # predicted_action, predicted_probs = networks["policy_net"]._select_action_with_correction(
    #   state=state,
    #   beta=networks["beta_net"].forward,
    #   action=action
    # )
    # Off-Policy-TopK-Correction    
    predicted_action, predicted_probs = networks["policy_net"]._select_action_with_TopK_correction(
      state=state,
      beta=networks["beta_net"].forward,
      action=action,
      K=rl_params["K"],
    )

    expected_reward = networks["value_net"](state, predicted_probs).detach()
    networks["policy_net"].rewards.append(expected_reward.mean())

    # value loss
    value_loss = None
    
    # -----------------
    # func value update
    with torch.no_grad():
        next_action = networks["target_policy_net"](next_state)
        target_value = networks["target_value_net"](next_state, next_action.detach())
        # temporal difference
        expected_value = reward + (1.0 - done) * rl_params["gamma"] * target_value
        expected_value = torch.clamp(
            expected_value, rl_params["min_value"], rl_params["max_value"]
        )

    value = networks["value_net"](state, action)
    value_loss = torch.pow(value - expected_value.detach(), 2).mean()
    
    if learn:
        optimizer["value_optimizer"].zero_grad()
        value_loss.backward(retain_graph=True)
        optimizer["value_optimizer"].step()
    # -----------------
    # -----------------
    # func policy update
    
    if ((step % 10 == 0) & (step > 0)):
        policy_loss = rl_params["reinforce_topk_corr"](
            policy=networks["policy_net"],
            optimizer=optimizer["policy_optimizer"],
            learn=learn,
        )
        # -----------------
        del networks["policy_net"].rewards[:]
        del networks["policy_net"].saved_log_probs[:]

        print("step: ", step, "| value: ", value_loss.item(), "| policy", policy_loss.item())

        ## soft update value_network
        for target_param, param in zip(networks["target_value_net"].parameters(), networks["value_net"].parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - rl_params["soft_tau"]) + param.data * rl_params["soft_tau"]
            )
        ## soft update policy_network
        for target_param, param in zip(networks["target_policy_net"].parameters(), networks["policy_net"].parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - rl_params["soft_tau"]) + param.data * rl_params["soft_tau"]
            )
            
        losses = {
            "value": value_loss.item(),
            "policy": policy_loss.item(),
            "step": step
        }
        
        