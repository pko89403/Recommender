import pandas as pd 
import igraph as ig 
import numpy as np 

COOCCURENCE_THRESHOLD = 10
NEIGHBOR_SIZE = 20

artist_raw_df = pd.read_csv(
    "part-00000-534f403e-0d00-4feb-af7f-8688dcf9c0ba-c000.csv",
    delimiter=",",
    header=None,
    dtype=str
)

artist_raw_df.columns = ["src", "dst", "weight"]
artist_raw_df = artist_raw_df.astype({"weight": int})

# # 10번 이상 등장하지 않는 artists 제거
filtered_artists = artist_raw_df[artist_raw_df["weight"] > COOCCURENCE_THRESHOLD]


filtered_artists.to_csv("artist_filtered_count.csv", sep=",", header=True, index=False)
filtered_artists = pd.read_csv("artist_filtered_count.csv", sep=",")


# # src + dst unique list 생성
artists_src_list = filtered_artists["src"].unique()
artists_dst_list = filtered_artists["dst"].unique()
artists_list = np.unique(np.append(artists_src_list, artists_dst_list))


# 20 개의 Neighbor만 가지도록 필터링 한다
neighbor_threshold = [i for i in range(NEIGHBOR_SIZE)]
artists_filtered_neightbor = pd.DataFrame()
for idx, artist in enumerate(artists_list):
    for direction in ["src", "dst"]:
        neighbor_artists = filtered_artists[filtered_artists[direction] == artist]
        if(len(neighbor_artists) < 20):
            continue

        neighbor_artists = neighbor_artists.sort_values("weight", ignore_index=True, ascending=False).filter(items=neighbor_threshold, axis=0)

        if idx == 0:
            artists_filtered_neightbor = neighbor_artists
        else:
            artists_filtered_neightbor = artists_filtered_neightbor.append(neighbor_artists, ignore_index=True)


artists_filtered_neightbor.to_csv("artists_filtered_neightbor.csv", sep=",", header=True, index=False)
artists_filtered_neightbor = pd.read_csv("artists.csv", sep=",")

directed_graph = ig.Graph.DataFrame(artists_filtered_neightbor, directed=True)




