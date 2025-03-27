import pandas as pd 
import numpy as np 
import json 
import torch
import torch.nn.functional as F
from tqdm import tqdm 

SPLIT_SIZE = 100



hidden_emb = np.load("torchGAE/emb.npy")

with open("torchGAE/index2node.json", "r") as f:
    index2artist_id = json.load(f)


artist_id_to_nm = dict()
with open("torchGAE/artist_id_to_nm.csv", "r") as f:
    for idx, line in enumerate(f):
        if idx == 0:
            continue
        
        id_nm = line.split("|")
        artist_id_to_nm[str(id_nm[0])] = str("|".join(id_nm[1:]))


hidden_emb_tensor = torch.from_numpy(hidden_emb)

def compute_metric_topk(emb : torch.tensor = None, normalization: bool = True, pop_bias: float = 5., ):
    emb_dim = emb.shape[1]
    
    mass = emb[:, emb_dim-1:emb_dim]
    emb = emb[:, 0:emb_dim-1]
    if normalization:
        emb = F.normalize(emb, p=2.0, dim=1)
    

    emb_splits = torch.split(emb, SPLIT_SIZE, dim=0)
    mass_splits = torch.split(mass, SPLIT_SIZE, dim=0)
    adj_rec = None
    
    for idx, (emb_split, mass_split) in tqdm(enumerate(zip(emb_splits, mass_splits)), total=len(emb_splits)):

        mass = mass_splits[idx].repeat(1, emb.shape[0])
        
        tmp_adj_rec = torch.cdist(emb_split, emb)
        tmp_adj_rec = mass - torch.mul(pop_bias, tmp_adj_rec)
        
        
        if adj_rec is None:
            adj_rec = tmp_adj_rec
        else:
            adj_rec = torch.cat((adj_rec, tmp_adj_rec), dim=0)
        

    top_k_values, top_k_indices = torch.topk(adj_rec, k=10, dim=1)
    
    
    for idx, (top_k_indice, top_k_value) in enumerate(zip(top_k_indices, top_k_values)):
        seed_artist = artist_id_to_nm.get(index2artist_id.get(str(idx), None), None).strip()
        top_k_indice = top_k_indice.numpy()
        test = [artist_id_to_nm.get(index2artist_id.get(str(i), None), None).strip() for i in top_k_indice ]

        result = "seed artist : {seed_artist} -> [{test}]".format(seed_artist=seed_artist, test=", ".join(test))
        print(result)
        


tok_result = compute_metric_topk(hidden_emb_tensor)
