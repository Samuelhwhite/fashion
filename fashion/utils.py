import time
import pickle
import requests
import json
import functools
import numpy as np
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path

if os.getenv('USER') == 'zgubic':
    loc = Path('/Users/zgubic/Projects/fashion/')
elif os.getenv('USER') == 'yourusername':
    loc = Path('')
else:
    print('"loc" variable not defined.\nEdit the first few lines of fashion/utils.py '+
    'with your username.\nYour username can be found by typing "echo $USER" in your terminal.')


def timeit(f):
    def decorated(*args, **kwargs):
        t0 = time.time()
        r = f(*args, **kwargs)
        print(' --> {} took {:2.2f}s'.format(f.__name__, time.time() - t0))
        return r
    return decorated


def cache(compute):

    @functools.wraps(compute)
    def wrapper(*args, force=False, **kwargs):

        # determine the file we are looking for
        args_str = ''.join(['_'+str(a) for a in args])
        kwargs_str = ''.join(['_{}{}'.format(k, kwargs[k]) for k in kwargs])
        fname = 'cache_{}{}{}.pkl'.format(compute.__name__, args_str, kwargs_str)
        fpath = loc / 'data' / fname

        # check cache
        if fpath.exists() and not force:
            print('Loading from cache {}'.format(fpath))
            res = pickle.load(open(fpath, 'rb'))

        # otherwise compute and cache
        else:
            print('Cache {} not found, computing.'.format(fpath))
            res = compute(*args, **kwargs)
            pickle.dump(res, open(fpath, 'wb'))

        return res

    return wrapper


def get_api_key():
    with open(loc / 'data' / 'api_key.txt', 'r') as f:
        api_key = f.read()
    return api_key


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
