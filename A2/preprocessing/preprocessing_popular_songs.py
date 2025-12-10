import preprocessing_general as ppg
import pandas as pd

popular_songs = pd.read_csv("../data/song_data.csv")
# ppg.df_info(popular_songs)


print(popular_songs.isnull().any(axis=1).sum())

# del popular_songs[popular_songs.columns[0]]
idx = popular_songs.columns.get_loc("song_name")
print('Categorical column "song_name" is at index: ',idx)

popular_songs_features = ppg.drop_cols(popular_songs, 0)


popular_songs_features_train, popular_songs_features_test  =ppg.train_test_split(popular_songs_features)

