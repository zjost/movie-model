

import pandas as pd
from load_dataset1 import get_boxofficemojo_movies
from load_dataset2 import load_dataset2


def merge_data():

    mojo_df = get_boxofficemojo_movies()
    meta_df = load_dataset2()

    #mojo_df['release_date'] = mojo_df['release_date_wide']

    merged_df = mojo_df.merge(meta_df, on = 'title')
    #merged_df = mojo_df.merge(meta_df, on = ['release_date', 'director'])

    #print(merged_df.head(20))

    return merged_df




if __name__ == '__main__':
    merge_data()
