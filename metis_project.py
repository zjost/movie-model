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
from sklearn import cross_validation
from sklearn import ensemble
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
    d['genre_trans'] = d['genre_trans'].fillna('Missing')
    genre_arr = vec.fit_transform(d['genre_trans'])
    genre_df = pd.DataFrame(genre_arr.todense(), columns=vec.get_feature_names())
    genre_df.columns = map(lambda x: 'genre_' + x, genre_df.columns.tolist())
    d = pd.concat([d,genre_df],axis=1)
    return d

def get_var_importance(model, x_cols):
    feature_importance = model.feature_importances_
    var_importance = pd.concat([pd.DataFrame(x_cols), pd.DataFrame(feature_importance)], axis=1)
    var_importance.columns = ['variable', 'importance']
    var_importance.sort(columns='importance', axis=0, ascending=False, inplace=True, na_position='last')
    var_importance.reset_index(range(0, var_importance.shape[0]), drop=True, inplace=True)
    var_importance['normalized'] = var_importance['importance']/var_importance.ix[0, 'importance']
    return var_importance


def feature_log_gross(movie_df, feature_name):
    nmovie = movie_df.shape[0]
    feature_log_gross =np.zeros(nmovie)
    feature_nmovie= np.zeros(nmovie)

    for i in range(nmovie) :
        feature = movie_df[feature_name][i]
        year = movie_df['year'][i]
        if type(feature) == list:
            feature = feature[0]
        if feature:
            feature_same_gross= movie_df['worldwide_gross'][(movie_df[feature_name]==feature) &  (movie_df['year'] < year)].dropna()
            if feature_same_gross.notnull().count() > 0:
                feature_log_gross[i] = np.log10(feature_same_gross).mean()
                feature_nmovie[i] = feature_same_gross.size
            else:
                feature_log_gross[i] = np.nan
                feature_nmovie[i] = 0

        else:
            feature_log_gross[i] = np.nan
            feature_nmovie[i] = 0
    return pd.concat([pd.DataFrame(feature_log_gross),pd.DataFrame(feature_nmovie)],axis = 1)



def feature_pre_log_gross(movie_df, feature_name):
    nmovie = movie_df.shape[0]
    feature_log_gross = np.zeros(nmovie)
    feature_nmovie = np.zeros(nmovie)
    for i in range(nmovie):
        feature = movie_df[feature_name][i]
        year = movie_df['year'][i]
        if type(feature)==list:
            feature = feature[0]

        if feature:
            if year > year_min:
                feature_pre_gross= movie_df['worldwide_gross'][(movie_df[feature_name]==feature) &  (movie_df['year'] == year-1)].dropna()
                if feature_pre_gross.notnull().count() > 0:
                    feature_log_gross[i] = np.log10(feature_pre_gross).mean()
                    feature_nmovie[i] = feature_pre_gross.size
                else:
                    feature_log_gross[i] = np.nan
                    feature_nmovie[i] = 0
            else:
                feature_log_gross[i] = np.nan
                feature_nmovie[i] = 0
    return pd.concat([pd.DataFrame(feature_log_gross),pd.DataFrame(feature_nmovie)],axis = 1)

def year_pre_log_gross(movie_df):
    nmovie = movie_df.shape[0]
    feature_log_gross = np.zeros(nmovie)
    feature_nmovie = np.zeros(nmovie)
    for i in range(nmovie):
        year = movie_df['year'][i]

        if (year > year_min):
            feature_pre_gross= movie_df['worldwide_gross'][(movie_df['year'] == year-1)].dropna()
            if feature_pre_gross.notnull().count() > 0:
                feature_log_gross[i] = np.log10(feature_pre_gross).mean()
                feature_nmovie[i] = feature_pre_gross.size
            else:
                feature_log_gross[i] = np.nan
                feature_nmovie[i] = 0
        else:
            feature_log_gross[i] = np.nan
            feature_nmovie[i] = 0
    return pd.concat([pd.DataFrame(feature_log_gross),pd.DataFrame(feature_nmovie)],axis = 1)


if __name__ == '__main__':
    DATA_DIR = os.path.join('/Users/ikt306/Documents/training/metis/capitalone-pilotthree/project_1/data', 'boxofficemojo')
    movie_list1 = file_load(DATA_DIR)
    movies_df1 = pd.DataFrame(movie_list1)
    movies_df1['title'] = movies_df1['title'].apply(lambda x: x.strip())

    DATA_DIR = os.path.join('/Users/ikt306/Documents/training/metis/capitalone-pilotthree/project_1/data', 'metacritic')
    movie_list2 = file_load(DATA_DIR)
    movies_df2 = pd.DataFrame(movie_list2)
    movies_df2['title'] = movies_df2['title'].apply(lambda x: x.strip() if type(x) == str else x)

    left_col = ['opening_weekend_take',
     'domestic_gross',
     'production_budget',
     'release_date_wide',
     'title',
     'worldwide_gross',
     'year']
    right_col = [
    'director',
     'genre',
     'runtime_minutes',
     'studio',
     'title',
     'rating',
     'metascore']

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



    year_min = movies_df['year'].min()

    year_pre_log_gross = year_pre_log_gross(movies_df)

    colnames = movies_df.columns.values
    dir_log_gross = feature_log_gross(movies_df,'director')
    studio_log_gross = feature_log_gross(movies_df, 'studio')

    dir_pre_log_gross = feature_pre_log_gross(movies_df,'director')
    studio_pre_log_gross = feature_pre_log_gross(movies_df, 'studio')

    ## add new variables
    dir_log_gross.columns = ['director_log_avg_gross', 'director_log_avg_count']
    studio_log_gross.columns=['studio_log_avg_gross','studio_log_count']

    dir_pre_log_gross.columns = ['director_pre_log_avg_gross', 'director_pre_log_avg_count']
    studio_pre_log_gross.columns=['studio_pre_log_avg_gross','studio_pre_log_count']
    year_pre_log_gross.columns = ['year_pre_log_avg_gross', 'year_pre_log_avg_count']





    dir_log_gross['director_log_avg_gross'] = dir_log_gross['director_log_avg_gross'].fillna(dir_log_gross['director_log_avg_gross'].mean())
    dir_log_gross['director_log_avg_count'] = dir_log_gross['director_log_avg_count'].fillna(dir_log_gross['director_log_avg_count'].mean())
    studio_log_gross['studio_log_avg_gross'] = studio_log_gross['studio_log_avg_gross'].fillna(studio_log_gross['studio_log_avg_gross'].mean())
    studio_log_gross['studio_log_count'] = studio_log_gross['studio_log_count'].fillna(studio_log_gross['studio_log_count'].mean())
    dir_pre_log_gross['director_pre_log_avg_gross'] = dir_pre_log_gross['director_pre_log_avg_gross'].fillna(dir_pre_log_gross['director_pre_log_avg_gross'].mean())
    dir_pre_log_gross['director_pre_log_avg_count'] = dir_pre_log_gross['director_pre_log_avg_count'].fillna(dir_pre_log_gross['director_pre_log_avg_count'].mean())
    studio_pre_log_gross['studio_pre_log_avg_gross'] = studio_pre_log_gross['studio_pre_log_avg_gross'].fillna(studio_pre_log_gross['studio_pre_log_avg_gross'].mean())
    studio_pre_log_gross['studio_pre_log_count'] = studio_pre_log_gross['studio_pre_log_count'].fillna(studio_pre_log_gross['studio_pre_log_count'].mean())
    year_pre_log_gross['year_pre_log_avg_gross'] = year_pre_log_gross['year_pre_log_avg_gross'].fillna(year_pre_log_gross['year_pre_log_avg_gross'].mean())
    year_pre_log_gross['year_pre_log_avg_count'] = year_pre_log_gross['year_pre_log_avg_count'].fillna(year_pre_log_gross['year_pre_log_avg_count'].mean())

    movies_df = pd.concat([movies_df,dir_log_gross,studio_log_gross,dir_pre_log_gross,studio_pre_log_gross,year_pre_log_gross],axis = 1)

    col_jie = movies_df.columns.tolist()
    col_jie = col_jie[-10:]


    #                                      u'worldwide_gross',
    col = ['metascore',
    'opening_weekend_take',
    'domestic_gross',
                                   'production_budget',
                                   'worldwide_gross',
                                                'year',
                                     'runtime_minutes',
                                'title_word_count',
                             'director_Clint Eastwood',
                            'director_Joel Schumacher',
                          'director_Steven Soderbergh',
                           'director_Steven Spielberg',
                                'director_Woody Allen',
                         'studio_Buena Vista Pictures',
                            'studio_Columbia Pictures',
                     'studio_Fox Searchlight Pictures',
                                    'studio_IFC Films',
                                'studio_Miramax Films',
                           'studio_Paramount Pictures',
                       'studio_Sony Pictures Classics',
       'studio_Twentieth Century Fox Film Corporation',
                           'studio_Universal Pictures',
                        'studio_Warner Bros. Pictures',
                                            'rating_R',
                                            'rating_G',
                                        'rating_PG_13',
                                        'rating_NC_17',
                                           'rating_PG',
                                            'month_01',
                                            'month_02',
                                            'month_03',
                                            'month_04',
                                            'month_05',
                                            'month_06',
                                            'month_07',
                                            'month_08',
                                            'month_09',
                                            'month_10',
                                            'month_11',
                                            'month_12',
                                        'genre_action',
                                     'genre_adventure',
                                     'genre_animation',
                                     'genre_biography',
                                        'genre_comedy',
                                         'genre_crime',
                                   'genre_documentary',
                                         'genre_drama',
                                        'genre_family',
                                       'genre_fantasy',
                                            'genre_fi',
                                       'genre_history',
                                        'genre_horror',
                                         'genre_music',
                                       'genre_musical',
                                       'genre_mystery',
                                       'genre_romance',
                                           'genre_sci',
                                         'genre_sport',
                                      'genre_thriller',
                                           'genre_war',
                                       'genre_western']
    # movies_df = movies_df[col]
    # movies_df = movies_df[col + col_jie]
    # movies_df.to_csv('/Users/ikt306/Documents/training/metis/capitalone-pilotthree/project_1/movie_clean.csv', header=True, index=False)
    movies_df['log_production_budget'] = np.log(movies_df['production_budget'])
    movies_df['log_opening_weekend_take'] = np.log(movies_df['opening_weekend_take'])
    movies_df['log_domestic_gross'] = np.log(movies_df['domestic_gross'])


    # movies_df['log_bug'] = movies_df['log_production_budget'] < 14

    train, validation = cross_validation.train_test_split(movies_df, test_size=0.3, random_state=41)

    x_col = [
                                   'log_production_budget',
                                                'year',
                                     'runtime_minutes',
                                'title_word_count',
                             'director_Clint Eastwood',
                            'director_Joel Schumacher',
                          'director_Steven Soderbergh',
                           'director_Steven Spielberg',
                                'director_Woody Allen',
                         'studio_Buena Vista Pictures',
                            'studio_Columbia Pictures',
                     'studio_Fox Searchlight Pictures',
                                    'studio_IFC Films',
                                'studio_Miramax Films',
                           'studio_Paramount Pictures',
                       'studio_Sony Pictures Classics',
       'studio_Twentieth Century Fox Film Corporation',
                           'studio_Universal Pictures',
                        'studio_Warner Bros. Pictures',
                                            'rating_R',
                                            'rating_G',
                                        'rating_PG_13',
                                        'rating_NC_17',
                                           'rating_PG',
                                            'month_01',
                                            'month_02',
                                            'month_03',
                                            'month_04',
                                            'month_05',
                                            'month_06',
                                            'month_07',
                                            'month_08',
                                            'month_09',
                                            'month_10',
                                            'month_11',
                                            'month_12',
                                        'genre_action',
                                     'genre_adventure',
                                     'genre_animation',
                                     'genre_biography',
                                        'genre_comedy',
                                         'genre_crime',
                                   'genre_documentary',
                                         'genre_drama',
                                        'genre_family',
                                       'genre_fantasy',
                                            'genre_fi',
                                       'genre_history',
                                        'genre_horror',
                                         'genre_music',
                                       'genre_musical',
                                       'genre_mystery',
                                       'genre_romance',
                                           'genre_sci',
                                         'genre_sport',
                                      'genre_thriller',
                                           'genre_war',
                                       'genre_western'
                                    ]
    x_col = x_col + col_jie

    # x_col = x_col + ['log_opening_weekend_take']
    # x_col.remove('log_opening_weekend_take')
    # x_col = x_col + ['log_bug']
    train = train.dropna()
    x_train = train[x_col]
    y_train = train['log_domestic_gross']
    validation = validation.dropna()
    x_val = validation[x_col]
    y_val = validation['log_domestic_gross']

    for n in [100]: # 100, 200, 500
        for leaf in [20]: # 1, 5, 10
            for depth in [3]:
                clf = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=n, subsample=1.0, min_samples_leaf=leaf, min_weight_fraction_leaf=0.0, max_depth=depth, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False)
                clf.fit(x_train, y_train)
                print n, leaf, depth
                print 'train error', clf.score(x_train, y_train)
                print 'testing error', clf.score(x_val, y_val)

    plt.scatter(y_val, clf.predict(x_val))
    plt.plot(y_val, y_val)
    plt.xlim(0,22)
    plt.ylim(0,22)
    plt.show()

    good_col = get_var_importance(clf, x_col)
    good_col_list = good_col['variable'].tolist()

    for i in [5, 10, 20, 30, 40, 50, len(good_col_list)]:
        train = train.dropna()
        x_train = train[good_col_list[:i]]
        y_train = train['log_domestic_gross']
        validation = validation.dropna()
        x_val = validation[good_col_list[:i]]
        y_val = validation['log_domestic_gross']

        clf = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_leaf=20, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False)
        clf.fit(x_train, y_train)
        print i
        print 'train error', clf.score(x_train, y_train)
        print 'testing error', clf.score(x_val, y_val)






    # get_var_importance(clf, x_col).to_csv('/Users/ikt306/Documents/training/metis/capitalone-pilotthree/project_1/var_importance_no1stweek.csv', header=True, index=False)

    train = train.dropna()
    x_train = train[x_col]
    y_train = train['log_opening_weekend_take']
    validation = validation.dropna()
    x_val = validation[x_col]
    y_val = validation['log_opening_weekend_take']

    for n in [100, 200, 500]:
        for leaf in [1, 5, 10]:
            clf = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=n, subsample=1.0, min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False)
            clf.fit(x_train, y_train)
            print 'train error', clf.score(x_train, y_train)
            print 'testing error', clf.score(x_val, y_val)

############ predict metascore
    train = train.dropna()
    x_train = train[x_col]
    y_train = train['metascore']
    validation = validation.dropna()
    x_val = validation[x_col]
    y_val = validation['metascore']

    for n in [100, 200, 500]:
        for leaf in [1, 5, 10]:
            clf = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=n, subsample=1.0, min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False)
            clf.fit(x_train, y_train)
            print 'train error', clf.score(x_train, y_train)
            print 'testing error', clf.score(x_val, y_val)



    from sklearn.feature_selection import SelectKBest, chi2

    X_new = SelectKBest(chi2, k=2).fit_transform(x_train, y_train)


    x_col_short = ['production_budget',
'runtime_minutes',
'year',
'studio_IFC Films',
'title_word_count',
'month_12',
'genre_drama',
'genre_comedy',
'genre_romance',
'genre_action']


    import statsmodels.formula.api as smf




    mod = smf.ols(formula='log_domestic_gross ~ log_production_budget + runtime_minutes + year + title_word_count + month_12 + genre_drama + genre_comedy + genre_comedy + genre_romance + genre_action', data=train[x_col+['log_domestic_gross']])
    res = mod.fit()
    print res.summary()


################
    good_col_list
    'log_domestic_gross ~ log_production_budget + runtime_minutes + year + title_word_count + month_12 + genre_drama + genre_comedy + genre_comedy + genre_romance + genre_action'


    for i in [5, 10, 20, 30]: # i = 20
        '+'.join(good_col_list[:i])
        linear_data = train[good_col_list[:i]+['log_domestic_gross']]

        mod = smf.ols(formula='log_domestic_gross ~ '+' + '.join(good_col_list[:i]), data=linear_data)
        res = mod.fit()
        print res.summary()
