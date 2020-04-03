import numpy as np
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec
import argparse
import warnings
warnings.filterwarnings('ignore')

def main(edgelist_name, walk_length, vector_length, no_epochs, model_name):
  # Create a graph
  print('Loading graph from edgelist...')
  fh = open(edgelist_name, 'r')
  graph = nx.read_weighted_edgelist(fh, delimiter=',',nodetype = str)
  print('Done')

  # Create an adjacency matrix
  print('Creating adjacency matrix...')
  A = nx.adjacency_matrix(graph)
  A_array = A.toarray()
  A_cumsum = np.cumsum(A_array,axis=1)
  matrix_lengths = A.shape[0]
  print('Done')

  print('Creating dictionaries for row probabilities and matrix identities...')
  row_total_probabilities = [A[x].sum() for x in range(0,matrix_lengths)]
  items = list(range(matrix_lengths))

  probability_dictionary = {}
  for item, probability in zip(items, row_total_probabilities):
      probability_dictionary[item] = probability

  prod_store_dict = {}
  for item2, node in zip(items, graph.nodes()):
      prod_store_dict[item2] = node

  def transition(x):
      if probability_dictionary[x] == 0:
          return x
      else:
          y = np.random.uniform(high=probability_dictionary[x])
          z = np.argwhere(A_cumsum[x]>y)[0][0]
          return z

  print('Done')

  #size is how many walks, the range is how many steps in each walk
  print('Generating random walks - may take a few minutes if long walks are chosen...')
  random_walks = list(range(matrix_lengths)) + list(range(matrix_lengths))

  #change it so you do a walk for every node 
  # rather than randomly choosing nodes to walk with 
  row = random_walks

  for null in range(0,walk_length):
      next_node =  np.asarray([transition(oof) for oof in row])
      random_walks = np.vstack((random_walks, next_node))
      row = random_walks[:][-1]
      
  walk_list = random_walks.T.tolist()
  #str_walk_list = [list(map(str, walk)) for walk in walk_list]    
  str_walk_list = [[prod_store_dict[thing] for thing in walk] for walk in walk_list]   
  print('Done')

  #running Word2Vec
  print('Running word2vec...')
  model = Word2Vec(size = vector_length, window = 4, sg = 1, hs = 0,
                  negative = 10, # for negative sampling
                  alpha=0.03, min_alpha=0.0007,
                  seed = 14)
  model.build_vocab(str_walk_list, progress_per=2)
  model.train(str_walk_list, total_examples = model.corpus_count, epochs=no_epochs, report_delay=1)
  print('Done')

  # Save the model
  print('Saving model...')
  model.save('{}'.format(model_name))
  print('Done')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='train word2vec model on graph based on edge list')
  parser.add_argument('--edgelist', type=str, required=True, help='input edgelist')
  parser.add_argument('--walk', type=int, default=30, help='length of the random walks')
  parser.add_argument('--vector', type=int, default=10, help='length of the output vectors')
  parser.add_argument('--epochs', type=int, default=20, help='no of epochs in skip-graph training')
  parser.add_argument('--name', type=str, default='DeepWalk_embeddings.model', help='output file name')

  args = parser.parse_args()
  main(args.edgelist, args.walk, args.vector, args.epochs, args.name)


