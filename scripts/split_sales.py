import pandas as pd
import sys
sys.path.insert(0, '..')
import fashion.preprocessing as prep
from fashion import utils


# load sales 
print('Loading sales (may take a while)')
shops = prep.load_shops(utils.loc / 'data' / '20200120_filiali.csv')
sales1819 = prep.load_sales(utils.loc / 'data' / '20200120_sales1819.csv', shops)

# compute the years
print('Computing the years index (may take a while)')
idx = pd.DatetimeIndex(sales1819.Date.astype(str))
years = idx.year

# save the separate files
print('Saving 2018 sales')
sales18 = sales1819[years == 2018]
sales18.to_csv(utils.loc / 'data' / '20200120_sales18.csv', index=False)
print('Saving 2019 sales')
sales19 = sales1819[years == 2019]
sales19.to_csv(utils.loc / 'data' / '20200120_sales19.csv', index=False)

