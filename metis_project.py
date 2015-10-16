# -*- coding: utf-8 -*-
# python /Users/ikt306/Documents/training/metis/capitalone-pilotthree/project_1/movie_analysis.py

import os
import json
from pprint import pprint
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from math import log
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.feature_extraction.text import CountVectorizer
# %matplotlib inline
# import this # python zen



def file_load(DATA_DIR):
    file_nm_list = os.listdir(DATA_DIR)
    movie_list = []
    for target_file_name in file_nm_list:
        target_file_path = os.path.join(DATA_DIR, target_file_name)
        with open(target_file_path, 'r') as target_file:
            movie = json.load(target_file)
            if type(movie) != dict:
                continue
            movie_list.append(movie)
    return movie_list

def top_dummy_create(data, col, n):
    df = data.copy()
    df[col] = df[col].fillna('Missing')
    top_category = df[col].value_counts().index[:n]
    top_category_df = df[df[col].isin(top_category)]
    dummy = pd.get_dummies(top_category_df[col])
    dummy.columns = map(lambda x: col+ '_' + x, dummy.columns.tolist())
    df = pd.concat([df,dummy],axis=1)
    df[dummy.columns] = df[dummy.columns].fillna(value=0)
    return df

def flag_genre(data):
    d = data.copy()
    vec = CountVectorizer(min_df = 0.003, binary=True)
    d.genre = d.genre.fillna('Missing')
    genre_arr = vec.fit_transform(d.genre)
    genre_df = pd.DataFrame(genre_arr.todense(), columns=vec.get_feature_names())
    genre_df.columns = map(lambda x: 'genre_' + x, genre_df.columns.tolist())
    d = pd.concat([d,genre_df],axis=1)
    return d


if __name__ == '__main__':
    DATA_DIR = os.path.join('/Users/ikt306/Documents/training/metis/capitalone-pilotthree/project_1/data', 'boxofficemojo')
    movie_list1 = file_load(DATA_DIR)
    movies_df1 = pd.DataFrame(movie_list1)
    movies_df1['title'] = movies_df1['title'].apply(lambda x: x.strip())

    DATA_DIR = os.path.join('/Users/ikt306/Documents/training/metis/capitalone-pilotthree/project_1/data', 'metacritic')
    movie_list2 = file_load(DATA_DIR)
    movies_df2 = pd.DataFrame(movie_list2)
    movies_df2['title'] = movies_df2['title'].apply(lambda x: x.strip() if type(x) == str else x)

    left_col = [
     u'domestic_gross',
     u'mojo_slug',
     u'production_budget',
     u'release_date_wide',
     u'title',
     u'worldwide_gross',
     u'year']
    right_col = [
    u'director',
     u'genre',
     u'runtime_minutes',
     u'studio',
     u'title',
     u'rating']

    movies_df = pd.merge(movies_df1[left_col], movies_df2[right_col], on='title', how='inner')
    # title feature
    movies_df['title_word_count'] = movies_df['title'].apply(lambda x: len(x.split()))

    # dummify
    movies_df = top_dummy_create(movies_df, 'director', 5)
    movies_df = top_dummy_create(movies_df, 'studio', 10)
    col = movies_df.columns.tolist()
    movies_df = top_dummy_create(movies_df, 'rating', 19)
    movies_df['rating_G'] = movies_df[['rating_G', 'rating_TV-G']].max(axis=1)
    movies_df['rating_PG_13'] = movies_df[['rating_PG-13', 'rating_TV-G']].max(axis=1)
    movies_df['rating_NC_17'] = movies_df['rating_NC-17']
    movies_df['rating_PG'] = movies_df[['rating_PG', 'rating_TV-PG']].max(axis=1)
    col = col + ['rating_R', 'rating_G', 'rating_PG_13', 'rating_NC_17', 'rating_PG']
    movies_df = movies_df[col]

    # year month
    movies_df['month'] = movies_df['release_date_wide'].apply(lambda x: x[5:7] if (x<>None and x<>np.nan) else x)
    movies_df = top_dummy_create(movies_df, 'month', 13)

    # Genre
    movies_df['genre_trans'] = movies_df['genre'].apply(lambda x: ' '.join(x) if type(x) == list else x)
    movies_df = flag_genre(movies_df)
    
