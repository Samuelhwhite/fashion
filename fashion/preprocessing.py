import pandas as pd
import numpy as np
import pickle
import sys
import os
sys.path.insert(0, '..')
from fashion import utils


def load_shops(path=utils.loc / 'data' / '20200120_filiali.csv', extra_info=True):

    # load
    df = pd.read_csv(path, low_memory=False)

    # translate the columns
    new_columns = ['StoreKey', 'Franchise', 'StoreType', 'Outlet', 'ZipCode', 'City']
    df.rename(inplace=True, columns=dict(zip(df.columns, new_columns)))

    # filter outlets
    df = df[df.Outlet != 'H']
    df.drop(['Outlet'], axis=1, inplace=True)

    # franchise correct
    df.Franchise = df.Franchise.map({'I':True, ' ':False})

    # StoreType
    df.StoreType = df.StoreType.map({' ':'Store'})

    # add extra information
    if extra_info:
        df['NUniqueProductsSold'] = get_nproducts_in_shop(df)

    return df


def get_nproducts_in_shop(shops):

    # load or compute the number of products sold in each store
    fpath = utils.loc / 'data' / 'product_counts.pkl'
    if os.path.exists(fpath):
        print('Loading product counts in stores')
        prod_counts = pickle.load(open(fpath, 'rb'))

    else:
        print('Computing product counts in stores')
        # load and concatenate sales
        sales17 = load_sales(utils.loc / 'data' / '20200120_sales17.csv', shops)
        sales1819 = load_sales(utils.loc / 'data' / '20200120_sales1819.csv', shops)
        sales = pd.concat([sales17, sales1819], axis=0)

        gb = sales.groupby(by='StoreKey')
        counts = gb["EAN"].nunique()
        prod_counts = {sk:counts[sk] for sk in shops.StoreKey if sk in counts}
        pickle.dump(prod_counts, open(fpath, 'wb'))

    # return an array of counts
    res = np.zeros(len(shops), dtype=int)
    for i, sk in enumerate(shops.StoreKey):
        nprod = prod_counts[sk] if sk in prod_counts else 0
        res[i] = nprod

    return res


def load_sales(path=utils.loc / 'data' / '20200120_sales17.csv', shops_df=load_shops(extra_info=False), nrows=None):

    # load
    if nrows:
        df = pd.read_csv(path, nrows=nrows)
    else:
        df = pd.read_csv(path)

    # translate the columns
    new_columns = ['StoreKey', 'ReceiptKey', 'Date', 'Hour', 'EAN', 'Volume', 'NetIncome']
    df.rename(inplace=True, columns=dict(zip(df.columns, new_columns)))

    # remove the ones which belong to warehouses (and delete the column)
    df = df.merge(shops_df[['StoreKey', 'StoreType']], how='left', on='StoreKey')
    df = df[df.StoreType == 'Store']
    df.drop(['StoreType'], axis=1, inplace=True)

    # drop some specific items from sales17
    df = df[df.NetIncome != 20901.0]
    df = df[df.NetIncome != -20901.0]
    df = df[df.NetIncome != 9095.0]
    df = df[df.NetIncome != -9095.0]

    return df


def load_products(path=utils.loc / 'data' / '20200120_barcode.csv'):

    # load the 
    df = pd.read_csv(path, low_memory=False)

    # translate the columns
    new_columns = ['EAN', 'ProductID', 'ColorCode', 'ColorDescription',
                   'Size', 'Gender', 'ItemCode', 'ItemDescription',
                   'ProductCategory', 'Season', 'Year', 'ColorID',
                   'SizeID', 'OriginalListedPrice', 'StandardCostOfGoods',
                   'ReceivedItems', 'IntroductionPeriod', 'Lifecycle',
                   'PriceRange', 'FashionContent', 'ItemWeight', 'InnovationContent']
    df.rename(inplace=True, columns=dict(zip(df.columns, new_columns)))

    # remove duplicate values (or unnecessary ones)
    cols_rm=['ColorID']
    df.drop(cols_rm, axis=1, inplace=True)

    # reduce the duplicates in colours
    def colour_transform(s):
        for c in ['-', '/', '.']:
            s = s.replace(c, ' ')
        return s.lower().split()[0]
    cd = 'ColorDescription'
    df[cd] = df[cd].apply(lambda s: colour_transform(s))

    # and keep them to 50 most popular colours
    nkeep = 50
    populars = df[cd].value_counts().index[:nkeep]
    df[cd] = df[cd].apply(lambda s: s if s in populars else 'other')

    return df
    

def get_stores_in_sales_data():

    fname = utils.loc / 'data' / 'sales_stores.csv'
    if os.path.exists(fname):
        print('loading')
        sales_stores = pickle.load(open(fname, 'rb'))

    else:
        print('computing')
        sales17 = load_sales(utils.loc / 'data' / '20200120_sales17.csv')
        stores17 = sales17.StoreKey.unique()
        sales1819 = load_sales(utils.loc / 'data' / '20200120_sales1819.csv')
        stores1819 = sales1819.StoreKey.unique()

        sales_stores = set(stores17).union(set(stores1819))
        pickle.dump(sales_stores, open(fname, 'wb'))

    return sales_stores


    

def main():

    shops = load_shops()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(shops.NUniqueProductsSold, bins=100)
    ax.set_xlabel('N unique products sold')
    ax.set_ylabel('Number of stores')
    #plt.savefig('test.pdf')






if __name__ == '__main__':
    main()
