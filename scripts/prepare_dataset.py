import argparse
import os
import time
import pandas as pd
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

@timeit
def main():
    """
    Create a dataset where each line represents cumulative sales of a product
    (summed over colours and sizes) in a week in a particular store.
    """

    # parse the arguments
    parser = argparse.ArgumentParser(description='prepare the dataset')
    parser.add_argument('--year', type=str, default='17',
                        help='which sales year to summarise')
    parser.add_argument('--force', default=False, action='store_true',
                        help='overwrite the output file')
    args = parser.parse_args()

    # check re-run is needed
    infile = utils.loc / 'data' / '20200120_sales{}.csv'.format(args.year)
    outfile = utils.loc / 'data' / 'basic_20{}.csv'.format(args.year)
    if os.path.exists(outfile) and not args.force:
        print('Output file {} already exists. Use --force option to overwrite.'.format(outfile))
        exit()

    # load the dataframes
    @timeit
    def load_df():
        print('Loading dataframes')
        shops = prep.load_shops()
        prods = prep.load_products()
        sales = prep.load_sales(infile)
        return shops, prods, sales
    shops, prods, sales = load_df()

    # merge the shops and product information
    @timeit
    def merge(sales, df, on, gb, features):
        if isinstance(on, str):
            on = [on]
        if isinstance(gb, str):
            gb = [gb]
        keeps = list(set(features).union(set(on + gb))) # avoid duplicates
        return sales.merge(df[keeps], how='left', on=on)

    print('Merging shop information')
    shop_features = ['Franchise']
    shop_merge_on = 'StoreKey'
    shop_group_by = 'StoreKey'
    sales = merge(sales, shops, shop_merge_on, shop_group_by, shop_features)

    print('Merging product information')
    prod_features = ['Gender', 'Season', 'OriginalListedPrice']
    prod_merge_on = 'EAN' # exact identifier of the item, including colour and size
    prod_group_by = ['ProductID', 'ColorDescription'] # aggregate over colour and size
    sales = merge(sales, prods, prod_merge_on, prod_group_by, prod_features)

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
        features = ['Week', shop_group_by] + prod_group_by + shop_features + prod_features

        df = sales.groupby(features)[targets].sum()
        df.reset_index(level=df.index.names, inplace=True)

        return df
    df = aggregate(sales)

    # save the resulting dataset
    @timeit
    def save(df, outfile):
        print('Saving')
        df.to_csv(outfile, index=False)
    save(df, outfile)


if __name__ == '__main__':
    main()
