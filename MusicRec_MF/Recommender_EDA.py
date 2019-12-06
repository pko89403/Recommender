# Create Custom class to handle data sets for the recommendation data
from torch.utils import data
import pandas as pd
import copy

def dat2csv(src_path, dst_path, dat_name, csv_name):
    df = pd.read_csv(src_path + dat_name, sep='\s+')
    df.to_csv(dst_path + csv_name)

def shuffle_data(path, filename):
    df = pd.read_csv(path + filename)
    df = df.sample(frac = 1)
    df.to_csv(path + filename + ".csv", index =False)

def data_normalize(path, filename):
    df = pd.read_csv(csv_path + filename).iloc[:, 1:]

    print( df.head() )

    unique_users = df.loc[:, "userID"].nunique()
    print(f"unique_users : {unique_users}")

    unique_artists = df.loc[:, "artistID"].nunique()
    print(f"unique_artists : {unique_artists}")

    total_entries = df.shape[0]
    print(f"total_entries : {total_entries}")
    print(f"total_entries / ( unique_users * unique_artists) : {total_entries / ( unique_users * unique_artists)}")
    
    grouped_df = df.groupby('artistID', as_index=False).count().iloc[:, :2].sort_values('userID', ascending=False)
    print(f"grouped_df : \n{grouped_df}")
    print(f"grouped_df.shape : {grouped_df.shape}")

    grouped_df = grouped_df[ grouped_df.userID > 1 ]
    print(f"grouped_df.shape useID > 1 : {grouped_df.shape}")
    grouped_df.shape

    print(f"find maximum artistID : \n{df.sort_values('artistID', ascending=False).iloc[1, :]}")
 
    df_scaled = copy.deepcopy(df)
    print(df_scaled.head())

    df_scaled.loc[:, ["weight"]] = df_scaled.weight.astype('float64')
    print(f"df_scaled.dtypes : {df_scaled.dtypes}")

    mean = df_scaled.weight.mean()
    std = df_scaled.weight.std()
    print(f"feature scaling result Mean : {mean} , Std : {std}")

    df_scaled[["weight"]] = df_scaled["weight"].apply(lambda x: (x - mean) / std )
    print(f"df_scaled order by weights \n{df_scaled.sort_values('weight').tail()}")
    df_scaled.to_csv(path + "normalized_" + filename)
    
    grouped_df = df.groupby('artistID', as_index=False).count().iloc[:, :2].sort_values('userID', ascending=False)
    grouped_df = grouped_df.loc[grouped_df["userID"] >= 5]
    grouped_df.columns = ['artistID', 'num_ratings']
    filtered_df = df.merge(grouped_df, how='inner', on='artistID')

    # check number of unique user/artist IDs
    unique_users = filtered_df.loc[:, "userID"].nunique()
    unique_artists = filtered_df.loc[:, "artistID"].nunique()
    total_entries = filtered_df.shape[0]

    # normalize the data within the new filtered set
    mean = filtered_df.weight.mean()
    std = filtered_df.weight.std()
    filtered_df[["weight"]] = filtered_df["weight"].apply(lambda x: (x-mean)/std)

    mean = df.weight.mean()
    std = df.weight.std()
    df[["weight"]] = df["weight"].apply(lambda x: (x-mean)/std)

    # remove the num_ratings column from the data frame because it is not needed
    filtered_df = filtered_df.iloc[:, :-1]

    # send the current scaled and filtered model to a csv file
    filtered_df.to_csv(path + "filtscale_" + filename)

    filtered_df = pd.read_csv(path + "filtscale_" + filename).iloc[:, 1:]
    
    contig_df = copy.deepcopy(filtered_df)
    unique_users = contig_df.userID.unique()
    user_to_index_dict = {o: i for i, o in enumerate(unique_users)}
    index_to_user_dict = {i: o for i, o in enumerate(unique_users)}
    contig_df.userID = contig_df.userID.apply(lambda x: user_to_index_dict[x])
    unique_artists = contig_df.artistID.unique()
    artist_to_index_dict = {o: i for i, o in enumerate(unique_artists)}
    index_to_artist_dict = {i: o for i, o in enumerate(unique_artists)}
    contig_df.artistID = contig_df.artistID.apply(lambda x: artist_to_index_dict[x])
    
    # write user/artists dictionaries to files so that the mappings can be used later when studying the resulting model
    import pickle
    pickle.dump(user_to_index_dict, open(path + "user_to_index_dict.txt", "wb"))
    pickle.dump(index_to_user_dict, open(path + "index_to_user_dict.txt", "wb"))
    pickle.dump(artist_to_index_dict, open(path + "artist_to_index_dict.txt", "wb"))
    pickle.dump(index_to_artist_dict, open(path + "index_to_artist_dict.txt", "wb"))

    # randomize order of the df and write to a csv
    contig_df = contig_df.sample(frac=1)
    contig_df.to_csv(path + "contig_" + filename, index=False)

if __name__ == "__main__":
    dat_path = "/Users/amore/Recsys_test/MusicRec_MF/Dataset/dat/"
    csv_path = "/Users/amore/Recsys_test/MusicRec_MF/Dataset/csv/"

    # dat2csv(dat_path, csv_path, "user_artists.dat", "music_ratings.csv")
    data_normalize(csv_path, "music_ratings.csv")
