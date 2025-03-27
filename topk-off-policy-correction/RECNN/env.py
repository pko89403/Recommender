import os
import pickle
from typing import List, Dict

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def try_progress_apply(dataframe: pd.DataFrame, function):
    return dataframe.apply(function)


class DataFuncKwargs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def keys(self):
        return self.kwargs.keys()
    
    def get(self, name: str):
        if name not in self.kwargs:
            raise AttributeError(
                "No kwargs with name {} found!".format(name)
            )
        return self.kwargs[name]
    
    def set(self, name: str, value):
        self.kwargs[name] = value

class DataFuncArgsMut:
    def __init__(
        self, df, base, users: List[int], user_dict: Dict[int, Dict[str, np.ndarray]]
    ):
        self.base = base
        self.users = users
        self.user_dict = user_dict
        self.df = df


def prepare_dataset(args_mut: DataFuncArgsMut, kwargs: DataFuncKwargs):

    # get args
    frame_size = kwargs.get("frame_size")
    key_to_id = args_mut.base.key_to_id
    df = args_mut.df
    
    df["rating"] = try_progress_apply(df["rating"], lambda i: 2 * (i - 2.5))
    df["movieId"] = try_progress_apply(df["movieId"], key_to_id)
    users = df[["userId", "movieId"]].groupby(["userId"]).size()
    users = users[users > frame_size].sort_values(ascending=False).index

    ratings = (
        df.sort_values(by="timestamp")
        .set_index("userId")
        .drop("timestamp", axis=1)
        .groupby("userId")
    )
    
    user_dict = {}
    
    def app(x):
        userid = x.index[0]
        user_dict[userid] = {}
        user_dict[userid]["items"] = x["movieId"].values
        user_dict[userid]["ratings"] = x["ratings"].values
        
    try_progress_apply(ratings, app)
    
    args_mut.user_dict = user_dict
    args_mut.users = users
    
    return args_mut, kwargs




class EnvBase:
    def __init__(self):
        self.train_user_dataset = None
        self.test_user_dataset = None
        self.embeddings = None
        self.key_to_id = None
        self.id_to_key = None

class DataPath:
    def __init__(
        self,
        base: str,
        ratings: str,
        embeddings: str,
        cache: str = "",
        use_cache: bool = True,
    ):
        self.ratings = base + ratings
        self.embeddings = base + embeddings
        self.cache = base + cache
        self.use_cache = use_cache


def batch_tensor_embeddings(batch, item_embeddings_tensor, frame_size, *args, **kwargs):
    def get_irsu(batch):
        items_t, ratings_t, sizes_t, users_t = (
            batch["items"],
            batch["ratings"],
            batch["sizes"],
            batch["users"],
        )
        return items_t, ratings_t, sizes_t, users_t
    
    
    items_t, ratings_t, sizes_t, users_t = get_irsu(batch)
    items_emb = item_embeddings_tensor[items_t.long()]
    b_size = ratings_t.size(0)
    
    items = items_emb[:, :-1, :].view(b_size, -1)
    next_items = items_emb[:, 1:, :].view(b_size, -1)
    ratings = ratings_t[:, :-1]
    next_ratings = ratings_t[:, 1:]
    
    state = torch.cat([items, ratings],1)
    next_state = torch.cat([next_items, next_ratings], 1)
    action = items_emb[:, -1, :]
    reward = ratings_t[:, -1]
    
    done = torch.zeros(b_size)
    done[torch.cumsum(sizes_t - frame_size, dim=0) - 1] = 1
    
    batch = {
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "done": done,
        "meta": {"users": users_t, "sizes": sizes_t},
    }
    return batch

def make_items_tensor(items_embeddings_key_dict):
    keys = list(sorted(items_embeddings_key_dict.keys()))
    key_to_id = dict(zip(keys, range(len(keys))))
    id_to_key = dict(zip(range(len(keys)), keys))
    
    items_embeddings_id_dict = {}
    for k in items_embeddings_key_dict.keys():
        items_embeddings_id_dict[key_to_id[k]] = items_embeddings_key_dict[k]
    items_embeddings_tensor = torch.stack(
        [items_embeddings_id_dict[i] for i in range(len(items_embeddings_id_dict))]
    )
    return items_embeddings_tensor, key_to_id, id_to_key

def sort_users_itemwise(user_dict, users):
    return (
        pd.Series(dict([(i, user_dict[i]["items"].shape[0]) for i in users]))
        .sort_values(ascending=False)
        .index
    )

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
    

class Env:
    def __init__(
        self,
        path: DataPath,
        prepare_dataset=prepare_dataset,
        embed_batch=batch_tensor_embeddings,
        **kwargs,
    ):
        self.base = EnvBase()
        self.embed_batch = embed_batch
        self.prepare_dataset = prepare_dataset
        if path.use_cache and os.path.isfile(path.cache):
            self.load_env(path.cache)
        else:
            self.process_env(path)
            if path.use_cache:
                self.save_env(path.cache)

    def load_env(self, where: str):
        self.base = pickle.load(open(where, "rb"))

    def save_env(self, where: str):
        pickle.dump(self.base, open(where, "wb"))

    def process_env(self, path: DataPath, **kwargs):
        if "frame_size" in kwargs.keys():
            frame_size = kwargs["frame_size"]
        else:
            frame_size = 10
            
        if "test_size" in kwargs.keys():
            test_size = kwargs["test_size"]
        else:
            test_size = 0.05
            
        movie_embeddings_key_dict = pickle.load(open(path.embeddings, "rb"))
        (
            self.base.embeddings,
            self.base.key_to_id,
            self.base.id_to_key,
        ) = make_items_tensor(movie_embeddings_key_dict)
        ratings = pd.read_csv(path.ratings)
        
        process_kwargs = DataFuncKwargs(
            frame_size=frame_size,
        )
        
        process_args_mut = DataFuncArgsMut(
            df=ratings,
            base=self.base,
            users=None,     # will be set later
            user_dict=None, # will be set later
        )
        
        self.prepare_dataset(process_args_mut, process_kwargs)
        self.base = process_args_mut.base
        self.df = process_args_mut.df
        users = process_args_mut.users
        user_dict = process_args_mut.user_dict
        
        train_users, test_users = train_test_split(users, test_size=test_size)
        train_users = sort_users_itemwise(user_dict, train_users)[2:]
        test_users = sort_users_itemwise(user_dict, test_users)
        self.base.train_user_dataset = UserDataset(train_users, user_dict)
        self.base.test_user_dataset = UserDataset(test_users, user_dict)

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides= a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def prepare_batch_static_size(
    batch, item_embeddings_tensor, frame_size=10, embed_batch=batch_tensor_embeddings
):
    item_t, ratings_t, sizes_t, users_t = [], [], [], []
    for i in range(len(batch)):
        item_t.append(batch[i]["items"])
        ratings_t.append(batch[i]["rates"])
        sizes_t.append(batch[i]["sizes"])
        users_t.append(batch[i]["users"])
        
    item_t = np.concatenate([rolling_window(i, frame_size + 1) for i in item_t], 0)
    ratings_t = np.concatenate(
        [rolling_window(i, frame_size + 1) for i in ratings_t], 0
    )
    
    item_t = torch.tensor(item_t)
    users_t = torch.tensor(users_t)
    ratings_t = torch.tensor(ratings_t).float()
    sizes_t = torch.tensor(sizes_t)
    
    batch = {"items": item_t, "users": users_t, "ratings": ratings_t, "sizes": sizes_t}

    return embed_batch(
        batch=batch,
        item_embeddings_tensor=item_embeddings_tensor,
        frame_size=frame_size,
    )

class FrameEnv(Env):
    def __init__(
        self,
        path,
        frame_size=10,
        batch_size=25,
        num_workers=1,
        *args,
        **kwargs,
    ):
        
        kwargs["frame_size"] = frame_size
        super(FrameEnv, self).__init__(
            path, min_seq_size=frame_size + 1, *args, **kwargs,
        )
        
        self.frame_size = frame_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_dataloader = DataLoader(
            self.base.train_user_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.prepare_batch_wrapper,
        )
        
        self.test_dataloader = DataLoader(
            self.base.test_user_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.prepare_batch_wrapper,
        )
    
    def prepare_batch_wrapper(self, x):
        batch = prepare_batch_static_size(
            x,
            self.base.embeddings,
            embed_batch=self.embed_batch,
            frame_size=self.frame_size,
        )
        return batch
    
    def train_batch(self):
        return next(iter(self.train_dataloader))
    
    def test_batch(self):
        return next(iter(self.test_dataloader))
