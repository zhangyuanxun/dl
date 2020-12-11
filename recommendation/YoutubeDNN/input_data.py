import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
import random
import tensorflow as tf
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

pd.set_option('display.max_columns', None)

def load_movielens_dataset():
    data_path = '../datasets/movielens/'

    # Load each data set (users, movies, and ratings).
    users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv(data_path + 'ml-100k/u.user', sep='|',
                        names=users_cols, encoding='latin-1')

    ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv( data_path + 'ml-100k/u.data', sep='\t',
                           names=ratings_cols, encoding='latin-1')

    # The movies file contains a binary feature for each genre.
    genre_cols = [
        "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]

    movies_cols = ['movie_id', 'title', 'release_date', "video_release_date", "imdb_url"] + genre_cols

    movies = pd.read_csv(data_path + 'ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')

    # Since the ids start at 1, we shift them to start at 0.
    users["user_id"] = users["user_id"].apply(lambda x: str(x - 1))
    movies["movie_id"] = movies["movie_id"].apply(lambda x: str(x - 1))
    movies["year"] = movies['release_date'].apply(lambda x: str(x).split('-')[-1])
    ratings["movie_id"] = ratings["movie_id"].apply(lambda x: str(x - 1))
    ratings["user_id"] = ratings["user_id"].apply(lambda x: str(x - 1))
    ratings["rating"] = ratings["rating"].apply(lambda x: float(x))

    # build features
    all_genres = movies[genre_cols].sum().to_dict()

    # genre labels
    genres_labels = {x: i for i, x in enumerate(genre_cols)}

    # add a new columns for all genre of a movie
    def get_all_genres(genres):
        active = [str(genres_labels[genre]) for genre, g in zip(genre_cols, genres) if g == 1]
        if len(active) == 0:
            return '0'
        return ','.join(active)

    movies['all_genres'] = [
        get_all_genres(gs) for gs in zip(*[movies[genre] for genre in genre_cols])]

    ratings = ratings.merge(movies, on='movie_id').merge(users, on='user_id')
    ratings['user_id'] = ratings['user_id'].astype(int)
    ratings['movie_id'] = ratings['movie_id'].astype(int)
    ratings = ratings.set_index(['user_id', 'unix_timestamp']).sort_index()
    ratings = ratings.reset_index()

    ratings['movie_type'] = np.where(ratings['rating'] >= 3, 'like', 'dislike')
    ratings['movie_name'] = ratings['title'].str[:-6]

    user_ids = ratings["user_id"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}

    movie_ids = ratings["movie_id"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

    title_ids = ratings["movie_name"].unique().tolist()
    title2title_encoded = {x: i for i, x in enumerate(title_ids)}
    title_encoded2title = {i: x for i, x in enumerate(title_ids)}

    ratings["user"] = ratings["user_id"].map(user2user_encoded)
    ratings["movie"] = ratings["movie_id"].map(movie2movie_encoded)
    ratings["title_d"] = ratings["movie_name"].map(title2title_encoded)

    sample_data = ratings[['user', 'occupation', 'sex']]
    sample_data = sample_data.reset_index()

    movie_list = ratings.groupby(['user', 'movie_type'])['movie'].apply(list).reset_index()
    title_list = ratings.groupby(['user'])['title_d'].apply(list).reset_index()
    genre_list = ratings.groupby(['user'])['all_genres'].unique().apply(list).reset_index()

    # Get the unique set of genre for all the users
    genre_list['all_genres'] = genre_list['all_genres'].apply(lambda x: list(set(','.join(x))))
    genre_list['all_genres'] = genre_list['all_genres'].apply(lambda x: [x for x in x if x.isdigit()])

    user_video_list = movie_list.pivot(index='user', columns='movie_type', values='movie').reset_index()
    user_video_list.fillna(ratings["movie"].max() + 1, inplace=True)

    sample_data = sample_data.drop('index', axis=1)
    sample_data = sample_data.drop_duplicates()

    user_final_list = pd.merge(user_video_list, title_list, how='left')
    user_title_list1 = pd.merge(user_final_list, genre_list, how='left')
    user_title_list = pd.merge(user_title_list1, sample_data, how='left')

    user_title_list['like'] = user_title_list['like'].apply(lambda x: x if type(x) is list else [x])
    user_title_list['dislike'] = user_title_list['dislike'].apply(lambda x: x if type(x) is list else [x])
    user_title_list['predict_labels'] = user_title_list['like'].apply(lambda x: (x[-1]))
    user_title_list['like'] = user_title_list['like'].apply(lambda x: (x[:-1]))

    print(pd.DataFrame(user_title_list[['user', 'dislike', 'like', 'title_d', 'all_genres', 'predict_labels']]).head(4))

    user_title_list_e = user_title_list[(user_title_list.user >= 1) &
                                        (user_title_list.user <= 500)]
    print(user_title_list.shape)


def build_model_inputs(dataset, user_profile, seq_max_len):
    uid = np.array([line[0] for line in dataset])  # user id
    seq = [line[1] for line in dataset]
    iid = np.array([line[2] for line in dataset])  # item id
    labels = np.array([line[3] for line in dataset])
    hist_len = np.array([line[4] for line in dataset])

    dataset_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=seq_max_len, padding='post',
                                                                    truncating='post', value=0)

    inputs = {"user_id": uid, "movie_id": iid, "hist_movie_id": dataset_seq_pad,
              "hist_len": hist_len}

    for key in ["gender", "age", "occupation", "zip"]:
        inputs[key] = user_profile.loc[inputs['user_id']][key].values

    return inputs, labels


if __name__ == '__main__':

    # load movielens dataset
    dataset = load_movielens_dataset()
