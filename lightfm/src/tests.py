import pickle5 as pickle
from tqdm import tqdm 
from pprint import pprint
import numpy as np
from lightfm import LightFM
import pandas as pd

data = dict()

with open("./traindata-mcpgpu.pickle", "rb") as f:
    data = pickle.load(f)
    
model : LightFM = None
with open("./model-mcpgpu.pickle", "rb") as f:
    model = pickle.load(f)
    
episodes_cols = ["episode_id", "episode_nm", "episode_type", "program_nm"]
# episodes = pd.read_csv("../datasets/episodes.csv", sep='\t', header=None)

tmp = dict()
for c in episodes_cols: 
    tmp[c] = []
    
with open("../datasets/episodes.csv") as epi_f:
    for line in epi_f:
        d = line.split('\t')
        for i, v in enumerate(d):
            tmp[episodes_cols[i]].append(str(v))
            
episodes = pd.DataFrame(tmp)
# episodes.columns = episodes_cols

"""
    item_embeddings : ( item_index, num_components )
    dot : (item_index, num_components) @ (num_components, 1) -> (item_index, 1)
"""
item_biases, item_embeddings = model.get_item_representations()




"""
item_id_map 은 key(item_id) value(item_embedding_index) 이다. 
"""
item_index_to_item_id = {v:k for k, v in data["item_id_map"].items()}
def proc_episode_meta(input: list):
    try:
        res = ", ".join([str(x) for x in input[0]])
    except:
        res = None
    return res 
    
def find_similar_episodes(res_dict: dict, episode_id: str ,num_k: int = 30):
    base_epi = episodes[episodes["episode_id"] == str(episode_id)].values.tolist()
    base_epi = proc_episode_meta(base_epi)
    try:
        res_dict["base"].append(base_epi)
    except:
        res_dict["base"] = [base_epi]
    
    item_idx = data["item_id_map"][int(episode_id)]
    
    scores = item_embeddings.dot(item_embeddings[item_idx])
    item_norms = np.linalg.norm(item_embeddings, axis=1)
    scores /= item_norms
    
    n_candidates = min(num_k * 50, len(episodes.index))
    max_score_item_id = np.argpartition(scores, -n_candidates)[-n_candidates:][::-1]    
    similar = sorted(zip(max_score_item_id, scores[max_score_item_id] / item_norms[item_idx]), key=lambda x: -x[1])

    append_num = 1
    for item_idx, sim_score in similar:
        item_id = item_index_to_item_id[item_idx] 
        sim_epi = episodes[episodes["episode_id"] == str(item_id)].values.tolist()
        
        sim_epi_meta = proc_episode_meta(sim_epi)
        
        if sim_epi_meta == None:
            continue
        else:
            sim_epi_meta = sim_epi_meta.strip() + ", " + str(sim_score)
        
        try:
            res_dict[str(append_num)].append(sim_epi_meta)
        except:
            res_dict[str(append_num)] = [sim_epi_meta]
        finally:
            append_num = append_num + 1
        
        if append_num > num_k:
            break        

res_dict = dict()
for epi_id in tqdm(episodes["episode_id"].values.tolist()):
    find_similar_episodes(res_dict, episode_id=epi_id, num_k=10)


res_df = pd.DataFrame(res_dict)
res_df.to_csv("lightfm_simlar_episodes.test.csv", sep=',')
