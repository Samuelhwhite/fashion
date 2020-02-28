import matplotlib.pyplot as plt
from pprint import pprint
import sys
sys.path.insert(0, '..')
from fashion import utils
from fashion import preprocessing
import numpy as np

# %%


def find_sales(date, store, ean, df):
    return(len(df[(df['Date'] == date) and
                  (df['StoreKey'] == store) and
                  (df['EAN'] == ean)]))


def random_data_sampler(num_samples,
                        shops=load_shops(extra_info=False),
                        sales=load_sales(),
                        products=load_products()):

    Variables = [sales.Date.unique(),
                 shops.StoreKey.unique(),
                 products.EAN.unique()]

    Date, Store, EAN =(np.random.choice(x, num_samples) for x in Variables)

    df = pd.DataFrame(list(zip(Date, Store, EAN)),
                      columns=['Date', 'Store', 'EAN'])

    df['sales_count'] = df.apply(lambda row: find_sales(row['Date'],
                                                        row['Store'],
                                                        row['EAN'],
                                                        products), axis=1)

    return(df)
