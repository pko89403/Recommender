import os

import pickle5 as pickle
from tqdm import tqdm
import numpy as np
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import (
    auc_score,
    precision_at_k,
    recall_at_k,
)
import pandas as pd 

from dataset import EpisodeDataset

ARTIFACTS_PATH = "../artifacts"
RANDOM_SEED = 1004
NUM_THREADS = 1


def train(epochs=1000):
    def result_summary(model, epoch, test_sets):
        train_auc = auc_score(model,test_sets, num_threads=NUM_THREADS).mean()
        train_pre_at_k = precision_at_k(model, test_sets, num_threads=NUM_THREADS).mean()
        train_rec_at_k = recall_at_k(model, test_sets, num_threads=NUM_THREADS).mean()
        
        print(f"{epoch} // Training Set AUC : {train_auc}, Training Set Pre@K : {train_pre_at_k}, Training Set Rec@K : {train_rec_at_k}")
    
    episode_dataset = EpisodeDataset()
    dataset = episode_dataset.data_dict

    train_interaction, test_interaction = random_train_test_split(dataset["interactions"], random_state=RANDOM_SEED)
    train_weights, test_weights = random_train_test_split(dataset["weights"], random_state=RANDOM_SEED)
        
    with open(os.path.join(ARTIFACTS_PATH, "traindata.pickle"), "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)    

    model = LightFM(
        no_components=150,
        learning_rate=1e-3,
        loss='warp',
        random_state=RANDOM_SEED)
    
    for epoch in tqdm(range(epochs)):
        model.fit_partial(interactions=train_interaction,
                user_features=None,
                item_features=None,
                num_threads=NUM_THREADS,
                sample_weight=train_weights, 
                epochs=10,
                verbose=True)
        
        with open(os.path.join(ARTIFACTS_PATH, f"model-{epoch}.pickle"), "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)    
    
        result_summary(model, epoch, test_interaction)
        
    return model
    
def find_similar_episodes(res_dict: dict, episode_id: str ,num_k: int, episodes, item_embeddings, item_idx_to_item_id, item_id_to_item_idx):
    def proc_episode_meta(input: list):
        try:
            res = ", ".join([str(x) for x in input[0]])
        except:
            res = None
        return res 

    base_epi = proc_episode_meta(episodes[episodes["episode_id"] == str(episode_id)].values.tolist())
    
    try:
        res_dict["base"].append(base_epi)
    except:
        res_dict["base"] = [base_epi]
    
    item_idx = item_id_to_item_idx[episode_id]
    
    # Similarity Scoring
    scores = item_embeddings.dot(item_embeddings[item_idx])
    item_norms = np.linalg.norm(item_embeddings, axis=1)
    scores /= item_norms
    
    # Fetch Top K Similar list
    n_candidates = min(num_k * 50, len(episodes.index))
    max_score_item_id = np.argpartition(scores, -n_candidates)[-n_candidates:][::-1]
    similar = sorted(zip(max_score_item_id, scores[max_score_item_id] / item_norms[item_idx]), key=lambda x: -x[1])

    append_num = 1
    for item_idx, sim_score in similar:
        item_id = item_idx_to_item_id[item_idx] 
        sim_epi_meta = proc_episode_meta(episodes[episodes["episode_id"] == str(item_id)].values.tolist())
        
        if sim_epi_meta is None:
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
    return res_dict


def main():
    episode_dataset = EpisodeDataset()
    dataset = episode_dataset.data_dict

    episodes_raw = episode_dataset.get_episodes_raw()
    item_id_to_item_idx = dataset["item_id_map"]
    item_idx_to_item_id = {v:k for k,v in dataset["item_id_map"].items()}
    
    light_fm_model = train(epochs=1)
        
    _, item_embeddings = light_fm_model.get_item_representations()
    res_dict = dict()
    
    for epi_id in tqdm(episodes_raw["episode_id"].values.tolist()):
        res_dict = find_similar_episodes(
                    res_dict=res_dict, 
                    episode_id=epi_id, 
                    num_k=10,
                    episodes=episodes_raw,
                    item_embeddings=item_embeddings,
                    item_idx_to_item_id=item_idx_to_item_id,
                    item_id_to_item_idx=item_id_to_item_idx)
        
    res_df = pd.DataFrame(res_dict)
    res_df.to_csv("lightfm_simlar_episodes.csv", sep=',')

if __name__ == "__main__":
    main()
    
