import os

import pandas as pd 
from lightfm.data import Dataset


BASE_PATH = "../datasets"
RATINGS = "ratings.csv"
EPISODES = "episodes.csv"


class EpisodeDataset:
    def __init__(self):
        episodes_cols = ["episode_id", "episode_nm", "episode_type", "program_nm"]
        self.episodes = self.read_csv(os.path.join(BASE_PATH, EPISODES), sep='\t', cols=episodes_cols)
        
        ratings_cols = ["charater_id", "program_id", "episode_id", "rating"]
        self.ratings = self.read_csv(os.path.join(BASE_PATH, RATINGS), sep=',', cols=ratings_cols)

        self.epi_features = self._set_episode_features(self.episodes, ["episode_id", "episode_type"])

        self.data_dict = self._genreate_dataset(user_feature=False, item_feature=False)

    def read_csv(self, path, sep, cols) -> pd.DataFrame:
        tmp = dict()
        for c in cols: 
            tmp[c] = []

        with open(path) as f:
            for line in f:
                d = line.split(sep)
                for i, v in enumerate(d):
                    tmp[cols[i]].append(str(v))
        return pd.DataFrame(tmp)                    

    def get_episodes_raw(self): 
        return self.episodes
    
    def get_ratings_raw(self):
        return self.ratings

    def _genreate_dataset(self, user_feature=False, item_feature=False):
        data_dict = dict()
        
        dataset = Dataset()
        
        if user_feature & item_feature:
            # dataset.fit(user_features=,
            #             item_features=,)
            pass
        elif user_feature  & ~item_feature:
            # dataset.fit(user_features=)
            pass 
        elif item_feature & ~user_feature:
            # dataset.fit(item_features=)
            pass        
        else:
            uniq_users = self.ratings["charater_id"].unique()
            uniq_items = self.ratings["episode_id"].unique()
            
            print("----- dataset statistics -----")
            print(f"unique users : {len(uniq_users)}")
            print(f"unique items : {len(uniq_items)}")
            
            dataset.fit(users=uniq_users,
                        items=uniq_items)
            
            
            rating_source = list()
            for d in self.ratings.values:
                if int(d[3]) >= 1:
                    rating_source.append((d[0], d[2], 1))
                else:
                    rating_source.append((d[0], d[2], -1))
            
            rating_source = [(d[0], d[2], int(d[3])) for d in self.ratings.values]
            
            print(f"total ratings : {len(rating_source)}")
            print("------------------------------")
            interactions, weights = dataset.build_interactions(rating_source)
            
        user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()

        data_dict["interactions"] = interactions
        data_dict["weights"] = weights
        data_dict["user_id_map"] = user_id_map
        data_dict["user_feature_map"] = user_feature_map
        data_dict["item_id_map"] = item_id_map
        data_dict["item_feature_map"] = item_feature_map
        
        return data_dict
    
    def _set_episode_features(self, episodes, features):
        return episodes[features]



