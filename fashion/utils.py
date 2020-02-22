import time
import pickle
import requests
import json
import numpy as np
import pandas as pd

import os
from pathlib import Path

if os.getenv('USER') == 'zgubic':
    loc = Path('/Users/zgubic/Projects/fashion/')
    state = 'mac'
elif os.getenv('USER') == 'yourusername':
    loc = Path('')
elif os.getenv('username') == 'odhra':
    loc = Path('')
    state = 'windows'
else:
    print('"loc" variable not defined.\nEdit the first few lines of fashion/utils.py '+
    'with your username.\nYour username can be found by typing "echo $USER" in your terminal.')




def get_api_key():
    with open(loc / 'data' / 'api_key.txt', 'r') as f:
        api_key = f.read()
    return api_key


def load_shops(path):
    
    # load
    df = pd.read_csv(path)
    
    # translate the columns
    new_columns = ['Store_Key', 'Franchise', 'Store_Type', 'Outlet', 'Zip_Code', 'City']
    df.rename(inplace=True, columns=dict(zip(df.columns, new_columns)))

    # filter outlets
    df = df[df.Outlet != 'H']
    df.drop(['Outlet'], axis=1, inplace=True)
    
    # filter warehouses
    df = df[df['Store_Type'] != 'P']
    df = df[df['Store_Type'] != 'S']
    df.drop(['Store_Type'], axis=1, inplace=True)
    
    return df


if state == 'mac':
    import geopandas as gpd

    def get_italian_geometry():
        precision = 10 # or 50 or 110
        fname = 'ne_{}m_admin_0_countries'.format(precision)
        fpath = loc / 'data' / '{}.shp'.format(fname)

        # download the data if not present locally
        if not os.path.exists(fpath):
            print('Data not found, downloading.')
            url = 'https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/{}.zip'.format(fname)
            os.system('wget {} -P {}'.format(url, loc / 'data'))
            os.system('unzip {d}/{n} -d {d}'.format(d=loc / 'data', n='{}.zip'.format(fname)))

        gdf = gpd.read_file(fpath)[['ADMIN', 'ADM0_A3', 'geometry']]
        gdf.columns = ['country', 'country_code', 'geometry']
        italy = gdf[gdf.country == 'Italy']['geometry'].iloc[0]

        return italy


def draw_italy(italy, ax):

    # draw line collection recursively
    def draw_line(line, ax):
        try:
            xs = [x for x, y in line.coords[:]]
            ys = [y for x, y in line.coords[:]]
            _ = ax.plot(xs, ys, color='k')
        except NotImplementedError:
            for l in line:
                draw_line(l, ax)

    # loop over polygons
    for polygon in italy:
        draw_line(polygon.boundary, ax)


def query_places(query, pagetoken=''):
    
    # send request
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json?"
    lat, lng = 41.28, 12.56 # roughly mid italy
    full_query = url + \
            'query=' + query + \
            '&key=' + get_api_key() + \
            '&locationbias=circle:10000@{},{}'.format(lat, lng) + \
            pagetoken
    req = requests.get(full_query)

    # only keep results near italy
    # (remove results in the UK due to IP address, should have been centred
    # in italy due to "locationbias", not sure why that doesnt work)
    def filter_italy(results):
        new_results = []
        for res in results:
            lng = res['geometry']['location']['lng']
            lat = res['geometry']['location']['lat']
            if lat > 35.48 and lat < 47.08 and lng > 6.60 and lng < 18.51:
                new_results.append(res)
        return new_results

    results = filter_italy(req.json()['results'])
    
    # if only one page given, return results
    if 'next_page_token' not in req.json().keys():
        return results
    
    # otherwise, dig deeper
    else:
        npt = '&pagetoken={}'.format(req.json()['next_page_token'])
        time.sleep(2)
        return results + query_places(query, npt)


def get_city_results():

    fpath = loc / 'data'/ 'city_results.pkl'

    # check if results already exist
    if os.path.exists(fpath):
        city_results = pickle.load(open(fpath, 'rb'))

    else:
        # query Google Places
        city_results = {}
        shops = load_shops(loc / 'data' / '20200120_filiali.csv')
        for city in shops.City:
            print(city)
            query = 'clothes shops near {}'.format(city)
            results = query_places(query)
            city_results[city] = results

        # dump the results
        pickle.dump(city_results, open(fpath, 'wb'))

    return city_results
