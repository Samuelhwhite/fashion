import pandas as pd
import sys
sys.path.insert(0, '..')
import fashion.preprocessing as prep
import fashion.utils as utils
import warnings
warnings.filterwarnings('ignore')
import argparse

def main(weeks, file_name, use_all):
  print('Loading data...')
  paths = [
    utils.loc / 'data' / '20200120_sales17.csv',
    utils.loc / 'data' / '20200120_sales1819.csv',
  ]
  if use_all:
    sales = pd.concat(prep.load_sales(path) for path in paths)
  else:
    # Default to 2017 data
    sales = prep.load_sales(paths[0])
  print('Done')

  print('Adding product and week columns to data...')
  sales['Product'] = sales.EAN.astype(str).str[:-3]
  sales['Week'] = pd.DatetimeIndex(sales.Date.astype(str)).week 
  print('Done')

  print('Generating product-2-store edgelist...')
  sales_per_store_sums = sales.groupby(['Product', 'StoreKey','Week'])['Volume'].sum() 
  first_n_weeks = sales_per_store_sums.groupby(['Product', 'StoreKey']).head(weeks)
  total_sales = first_n_weeks.groupby(['Product', 'StoreKey']).sum()
  max_val = total_sales.max()
  product2store_graph = total_sales.reset_index()
  product2store_graph.Volume = product2store_graph.Volume/max_val
  print('Done')

  print('Saving product-2-store edgelist...')
  product2store_graph.to_csv(r'{}'.format(file_name), header=False, index=False)
  print('Done')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description='generate edge-list representing a bipartite graph of networks and stores'
  )

  parser.add_argument('--weeks', type=int, default=2, help='number of weeks')
  parser.add_argument(
    '--all', default=False, action='store_true',
    help='whether to use all the data (default only 2017)'
  )
  parser.add_argument(
    '--name', type=str, default='list_of_edges.csv',
    help='file name of the output'
  )

  args = parser.parse_args()

  main(args.weeks, args.name, args.all)