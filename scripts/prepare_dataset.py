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
np.random.seed(2020)

@utils.timeit
@utils.cache
def aggregate(year):

    # load the dataframes
    infile = utils.loc / 'data' / '20200120_sales{}.csv'.format(year)

    @utils.timeit
    def load_df():
        print('Loading dataframes')
        sales = prep.load_sales(infile)
        return sales
    sales = load_df()

    # compute the week of the year
    @utils.timeit
    def week(sales):
        print('Computing week number from date')
        sales['Week'] = pd.DatetimeIndex(sales.Date.astype(str)).week
        return sales
    sales = week(sales)

    # group by and aggregate
    @utils.timeit
    def agg(sales):
        print('Grouping and aggregating')
        targets = ['Volume']
        features = ['EAN', 'Week', 'StoreKey']
        df = sales.groupby(features)[targets].sum()
        return df
    df = agg(sales)

    @utils.timeit
    def create_dict(df):
        print('Creating a dict from the df')
        d = df.to_dict(orient='index')
        return d
    d = create_dict(df)

    return d


@utils.timeit
def generate_skeleton(length, EANs, weeks, store_keys):

    print('Generating (EAN, week, StoreKey) combinations')

    # generate random examples and pack them in a hash table
    e, w, sk = (np.random.choice(array, length).reshape(length, 1) for array in [EANs, weeks, store_keys])
    array = np.hstack([e, w, sk, np.zeros([length, 1])])

    skeleton = pd.DataFrame(array, columns=['EAN', 'Week', 'StoreKey', 'Volume'])

    return skeleton


@utils.timeit
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


@utils.timeit
def merge(skeleton, df, features, on):
    print('Merging features')
    keeps = set(features).union(set([on]))
    return skeleton.merge(df[keeps], how='left', on=on)


@utils.timeit
@utils.cache
def sample(year, sample, force=False):

    print('Sampling sales dataset')
    sales = aggregate(year=year, force=force)

    # prepare EANs sold and store keys available in the year
    EANs, store_keys = (prep.unique_in_sales_data(c, year) for c in ['EAN', 'StoreKey'])
    weeks = [i for i in range(1, 53)]

    # sample random combinations
    skeleton = generate_skeleton(sample, EANs, weeks, store_keys)

    # fill with the existing sales
    skeleton = fill_skeleton(skeleton, sales)

    # merge shops features
    shops = prep.load_shops()
    shop_features = ['Franchise', 'NUniqueProductsSold', 'NTotalProductsSold']
    shop_on = 'StoreKey'
    skeleton = merge(skeleton, shops, shop_features, shop_on)

    # merge the product features
    prods = prep.load_products()
    prod_features = ['Gender', 'Season', 'OriginalListedPrice']
    prod_on = 'EAN'
    skeleton = merge(skeleton, prods, prod_features, prod_on)

    # hack, sorry
    # (AI season is not represented in 2018 sales data, does not create a column for categorical variables)
    if year == '18':
        skeleton.loc[0, 'Season'] = 'AI'

    return skeleton


@utils.timeit
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
    parser.add_argument('--sample', type=int, default=False, 
                        help='how many samples to generate when sampling from the dataset')
    args = parser.parse_args()

    if args.sample:
        s = sample(args.year, args.sample, force=args.force)


if __name__ == '__main__':
    main()
