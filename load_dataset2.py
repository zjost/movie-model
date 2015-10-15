'''
This function will import the metacritic data into a pandas df
and returns a pandas dataframe containing the data
'''

import pandas as pd
import json
import os
from pprint import pprint

def load_dataset2():
    HOME = os.getenv('HOME')
    DATA_DIR = os.path.join(HOME, 'capitalone-pilotthree','project_1', 'data', 'metacritic')

    file_list = os.listdir(DATA_DIR)

    #print(file_list)
    metacritic = []

    for filename in file_list:
        target_filepath = os.path.join(DATA_DIR, filename)
        with open (target_filepath) as target_file:
            movie = json.load(target_file)
            if type(movie) != dict:
                continue
            metacritic.append(movie)
            #print(json.load(target_file))

    metacritic_df = pd.DataFrame(metacritic)
    metacritic_df = metacritic_df[pd.notnull(metacritic_df['title'])]


    return metacritic_df
