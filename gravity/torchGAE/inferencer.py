import itertools
from multiprocessing import Pool
from multiprocessing import cpu_count
from functools import partial
import os

import numpy as np
import pandas as pd
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

class Inferencer:
    def __init__(self, config):
        self.artifact_path = config["inference"]["path"]["artifact"]
        self.emb_name = config["inference"]["path"]["embedding"]
        self.index2node_name = config["inference"]["path"]["index2node"]
        self.similar_artists_name = config["inference"]["path"]["similar_artists"]
        self.emb_split_size = config["inference"]["emb_split_size"]
        
        self.embedding = None
        self.index2node = None
        
        self.top_k_values = None 
        self.top_k_indices = None
        
        self.load_data()
        
    def load_data(self):
        # LOAD EMBEDDING 
        emb_path = os.path.join(
            self.artifact_path,
            self.emb_name
        )
        self.embedding = torch.from_numpy(
            np.load(emb_path)
        )

        # LOAD INDEX TO NODE_ID DICT
        index2node_path = os.path.join(
            self.artifact_path,
            self.index2node_name
        )
        with open(index2node_path, "r") as f:
            self.index2node = json.load(f)

    def process(self, chunks, params):
        result = []
        
        for chunk in chunks:
            try:
                seed_artist = self.index2node[str(chunk)].strip()
            except:
                print(chunks)
                
            top_k_index = self.top_k_indices[chunk].numpy()
            top_k_value = self.top_k_values[chunk].numpy()
            
            for idx, (top_index, top_value) in enumerate(zip(top_k_index, top_k_value)):
                if idx == 0:
                    continue
                
                try:
                    similar_artist = self.index2node[str(top_index)].strip()
                    
                    result.append(
                        {
                            "seed_artist" : seed_artist,
                            "simliar_artist" : similar_artist,
                            "similarity" : round(top_value, 5)
                        }
                    )
                except Exception as e:
                    print(f"similar_artist : {top_index},\tdetail : {e}", flush=True)
                    
        return result

    def parallel_proc(self, n_cores, total_artists):
        if cpu_count() < n_cores:
            raise ValueError("The numver of CPU's specified exceed the amount available")

        print(n_cores, total_artists)
        chunks = np.array_split([idx for idx in range(total_artists)], n_cores)
        
        func = self.process
        func_params = {
            "test": "test"
        }
        
        pool = Pool(n_cores)
        res = pool.map(
            partial(
                func,
                params=func_params
            ),
            chunks
        )
        pool.close()
        pool.join()
        
        result = list(itertools.chain.from_iterable(res))
        return pd.DataFrame.from_records(result)

    def topK(self, pop_bias: float=5., top_k: int=10, norm: bool=True):
        num_emb = self.embedding.shape[0]
        emb_dim = self.embedding.shape[1]
        
        mass = self.embedding[:, emb_dim-1:emb_dim]
        emb = self.embedding[:,0:emb_dim-1]
        
        if norm:
            emb = F.normalize(emb, p=2.0, dim=1)
        
        mass_splits = torch.split(mass, self.emb_split_size, dim=0)
        emb_splits = torch.split(emb, self.emb_split_size, dim=0)
        
        adj_rec = None
        for idx, (mass_split, emb_split) in tqdm(enumerate(zip(mass_splits, emb_splits)), total=len(emb_splits)):
            mass = mass_split.repeat(1, num_emb)
            
            tmp_dist = torch.cdist(emb_split, emb)
            tmp_adj_rec = mass - torch.mul(pop_bias, tmp_dist)

            if adj_rec is None:
                adj_rec = tmp_adj_rec
            else:
                adj_rec = torch.cat((adj_rec, tmp_adj_rec), dim=0)
        
        self.top_k_values, self.top_k_indices = torch.topk(
            adj_rec,
            k=(top_k+1),
            dim=1
        )
        
        result_df = self.parallel_proc(n_cores=10, total_artists=num_emb)
        
        result_df.to_csv(
            os.path.join(self.artifact_path, self.similar_artists_name)
        )
        
        