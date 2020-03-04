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
        for kind in ['Unique', 'Total']:
            df['N{}ProductsSold'.format(kind)] = get_nproducts_in_shop(df, kind)

    return df


def get_nproducts_in_shop(shops, kind):

    # load or compute the number of products sold in each store
    fpath = utils.loc / 'data' / '{}_product_counts.pkl'.format(kind)
    print(fpath)
    if os.path.exists(fpath):
        print('Loading {} product counts in stores'.format(kind))
        prod_counts = pickle.load(open(fpath, 'rb'))

    else:
        print('Computing {} product counts in stores'.format(kind))
        # load and concatenate sales
        sales = load_sales(utils.loc / 'data' / '20200120_sales17.csv', shops)
        if kind == 'Unique':
            sales1819 = load_sales(utils.loc / 'data' / '20200120_sales1819.csv', shops)
            sales = pd.concat([sales, sales1819], axis=0)

        gb = sales.groupby(by='StoreKey')
        if kind == 'Unique':
            counts = gb["EAN"].nunique()
        elif kind == 'Total':
            counts = gb["Volume"].sum()
            
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

    # keep only the most common sizes (such that 95% of items retain their size, other sizes will be marked as 'other')
    size = 'Size'
    cs = np.cumsum(df[size].value_counts())
    nkeep = np.sum(cs < 0.95 * cs.iloc[-1])
    populars = df[size].value_counts().index[:nkeep]
    df[size] = df[size].apply(lambda s: s if s in populars else 'other')

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


def get_EAN2pid():

    fpath = utils.loc / 'data'/ 'EAN2pid.pkl'

    # check if results already exist
    if os.path.exists(fpath):
        EAN2pid = pickle.load(open(fpath, 'rb'))

    else:
        print('EAN2pid dict does not exist yet, computing it now.')

        # compute the dict
        prods = load_products()

        EAN2pid = {}
        for i in range(len(prods)):
            row = prods.iloc[i]
            EAN2pid[row['EAN']] = row['ProductID']


        # dump the results
        pickle.dump(EAN2pid, open(fpath, 'wb'))

    return EAN2pid


def get_pid2EAN():

    fpath = utils.loc / 'data'/ 'pid2EAN.pkl'

    # check if results already exist
    if os.path.exists(fpath):
        pid2EAN = pickle.load(open(fpath, 'rb'))

    else:
        print('pid2EAN dict does not exist yet, computing it now.')

        # compute the dict
        EAN2pid = get_EAN2pid()
        pid2EAN = {EAN2pid[e]:e for e in EAN2pid}

        # dump the results
        pickle.dump(pid2EAN, open(fpath, 'wb'))

    return pid2EAN


def main():

    shops = load_shops()

    #import matplotlib.pyplot as plt
    #for var in ['NTotalProductsSold', 'NUniqueProductsSold']:
    #    fig, ax = plt.subplots()
    #    ax.hist(shops[var], bins=100)
    #    ax.set_xlabel(var)
    #    ax.set_ylabel('Number of stores')
    #    plt.savefig(utils.loc / 'figures' / '{}.pdf'.format(var))






if __name__ == '__main__':
    main()
