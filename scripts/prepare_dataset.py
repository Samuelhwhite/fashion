import argparse
import pickle
import os
import time
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import sys
sys.path.insert(0, '..')
import fashion.preprocessing as prep
from fashion import utils


def timeit(f):
    def decorated(*args, **kwargs):
        t0 = time.time()
        r = f(*args, **kwargs)
        print(' --> {} took {:2.2f}s'.format(f.__name__, time.time() - t0))
        return r
    return decorated


@utils.cache
def aggregate(year):

    # load the dataframes
    infile = utils.loc / 'data' / '20200120_sales{}.csv'.format(year)

    @timeit
    def load_df():
        print('Loading dataframes')
        shops = prep.load_shops()
        prods = prep.load_products()
        sales = prep.load_sales(infile, nrows=1000)
        return shops, prods, sales
    shops, prods, sales = load_df()

    ## merge the shops and product information
    #@timeit
    #def merge(sales, df, on, gb, features):
    #    if isinstance(on, str):
    #        on = [on]
    #    if isinstance(gb, str):
    #        gb = [gb]
    #    keeps = list(set(features).union(set(on + gb))) # avoid duplicates
    #    return sales.merge(df[keeps], how='left', on=on)

    #print('Merging shop information')
    #shop_features = ['Franchise', 'NUniqueProductsSold', 'NTotalProductsSold']
    #shop_merge_on = 'StoreKey'
    #shop_group_by = ['StoreKey']
    ##sales = merge(sales, shops, shop_merge_on, shop_group_by, shop_features)

    #print('Merging product information')
    #prod_features = ['Gender', 'Season', 'OriginalListedPrice']
    #prod_merge_on = 'EAN' # exact identifier of the item, including colour and size
    #prod_group_by = ['EAN' if args.EAN else 'ProductID'] # aggregate over colour and size
    ##sales = merge(sales, prods, prod_merge_on, prod_group_by, prod_features)

    # compute the week of the year
    @timeit
    def week(sales):
        print('Computing week number from date')
        sales['Week'] = pd.DatetimeIndex(sales.Date.astype(str)).week
        return sales
    sales = week(sales)

    # group by and aggregate
    @timeit
    def aggregate(sales):
        print('Grouping and aggregating')
        targets = ['Volume']
        features = ['EAN', 'Week', 'StoreKey']

        df = sales.groupby(features)[targets].sum()

        return df
    df = aggregate(sales)

    # hack, sorry
    # (AI season is not represented in 2018 sales data, does not create a column for categorical variables)
    #if args.year == '18':
    #    df.loc[0, 'Season'] = 'AI'

    @timeit
    def create_dict(df):
        print('Creating a dict from the df')
        d = df.to_dict(orient='index')
        return d
    d = create_dict(df.head(1000)) #TODO

    return d


@timeit
def generate_skeleton(length, EANs, weeks, store_keys):

    print('Generating (EAN, week, StoreKey) combinations')

    # generate random examples and pack them in a hash table
    e, w, sk = (np.random.choice(array, length).reshape(length, 1) for array in [EANs, weeks, store_keys])
    array = np.hstack([e, w, sk, np.zeros([length, 1])])

    skeleton = pd.DataFrame(array, columns=['EAN', 'Week', 'StoreKey', 'Volume'])

    return skeleton


@timeit
def fill_skeleton(skeleton, sales):

    tqdm.pandas()
    def change_volume(row, sales):
        e, w, sk = row['EAN'], row['Week'], row['StoreKey']
        try:
            vol = sales[(e, w, sk)]['Volume']
            return vol
        except KeyError:
            return 0
    volume = skeleton.progress_apply(lambda row: change_volume(row, sales), axis=1)
    skeleton['Volume'] = volume

    return skeleton


@timeit
def load_sales_dict(year):
    print('Loading sales dict for 20{}'.format(year))
    sales = pickle.load(open('../data/aggregate_20{}.pkl'.format(year), 'rb'))
    return sales


@timeit
def sample(year, sample):

    print('Sampling sales dataset')

    # prepare EANs sold and store keys available in the year
    EANs, store_keys = (prep.unique_in_sales_data(c, year) for c in ['EAN', 'StoreKey'])
    weeks = [i for i in range(1, 53)]

    # sample random combinations
    skeleton = generate_skeleton(sample, EANs, weeks, store_keys)

    # fill with the existing sales
    sales = load_sales_dict(year)
    skeleton = fill_skeleton(skeleton, sales)

    # merge the product and store information
    # TODO

    return skeleton

    

@timeit
def main():
    """
    Create a dataset where each line represents cumulative sales of a product
    in a week in a particular store.
    """

    # parse the arguments
    parser = argparse.ArgumentParser(description='prepare the dataset')
    parser.add_argument('--year', type=str, default='17',
                        help='which sales year to summarise')
    parser.add_argument('--force', default=False, action='store_true',
                        help='overwrite the output file')
    parser.add_argument('--aggregate', default=False, action='store_true',
                        help='aggregate the sales data and save as a dictionary')
    parser.add_argument('--sample', type=int, default=False, 
                        help='how many samples to generate when sampling from the dataset')
    args = parser.parse_args()

    if args.aggregate:
        aggregate(year=args.year, force=args.force)

    if args.sample:
        sample(year=args.year, sample=args.sample)


if __name__ == '__main__':
    main()
