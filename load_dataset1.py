"""loaddata is a module for accessing scraped data about movies from
BoxOfficeMojo and Metacritic.
It's built specifically to work with the data collected for the
CapitalOne Metis Data Science Python Bootcamp Pilot Extravaganza 2K15.
"""

# imports
import os
import json
import pprint
import pandas as pd

# constants
HOME = os.getenv('HOME')
DATA_DIR = os.path.join(HOME, 'capitalone-pilotthree','project_1', 'data')
MOJO_DIR = os.path.join(DATA_DIR, 'boxofficemojo')

def get_boxofficemojo_movies():
    file_contents = os.listdir(MOJO_DIR)

    movie_list = []

    for filename in file_contents:
        filepath = os.path.join(MOJO_DIR, filename)

        with open(filepath, 'r') as movie_file:
            movie_data = json.load(movie_file)

        movie_list.append(movie_data)

    #print "Parsed %i movies from %i files" % (len(movie_list),
                                            #len(file_contents))
    return pd.DataFrame(movie_list)

if __name__ == "__main__":
    movies = get_boxofficemojo_movies()
