import random
from random import sample
import networkx as nx
import os

# # tree generator
# def generate_random_tree(n):
#   parent = {}
#   nodes = list(range(n))
#   random.shuffle(nodes)
#   root = nodes[0]
#   parent[root] = root
#   for i in range(1, len(nodes)):
#     parent[nodes[i]] = nodes[random.randrange(i)]
#   return parent


# def get_node_depths(tree):
#     depths = {}  # size of tree, node:level
#     for node in tree:
#         level = 0
#         curr = node
#         while tree[curr] != curr:
#             if curr in depths:
#                 level += depths[curr]
#                 break  
#             curr = tree[curr]
#             level += 1
#         depths[node] = level
#     return depths

# def get_distance(tree, src, dest, depths):
#     s = src; d = dest
#     while s != d:
#         if depths[d] > depths[s]:
#             d = tree[d]
#         elif depths[s] > depths[d]:
#             s = tree[s]
#         else:
#             if s != d and depths[s] == depths[d]:
#                 d = tree[d]
#     # least common ancestor
#     lca = s
#     src_depth = depths[src] - depths[lca]
#     dest_depth = depths[dest] - depths[lca]
#     return (src_depth + dest_depth)


# src, dst, distance_label
def generate_trees_dataset(N):
    dataset = []     # node1, node2, distance
    all_trees = nx.Graph()
    counter = 0

    for n in range(3, N): # For each tree
        tr = nx.random_tree(n)
        tr = nx.convert_node_labels_to_integers(tr, first_label=counter, label_attribute='old_label')
        counter += n
        all_trees = nx.compose(all_trees, tr)

    all_trees = nx.convert_node_labels_to_integers(all_trees, first_label=0, label_attribute='old_label')
    all_shortest_paths = nx.shortest_path_length(all_trees)
    for all_paths in all_shortest_paths:
        src = all_paths[0]
        for dest in all_paths[1]:
            dataset.append([src, dest, all_paths[1][dest]])
    nx.draw(all_trees)

    return dataset

'''
train_pct: percentage of each graph to be used for training
'''
def generate_real_graphs_dataset(files_list):
    dataset = []
    max_node_id = 0
    G = nx.DiGraph()
    
    for file in files_list:
      if max_node_id > min([int(line.split()[0]) for line in open(file)]):
         # create a new file with the same name+unique id
         # replace all node ids with prev_id + max_node_id
        lines = []
        with open(file) as f:
          lines = f.readlines()
        filename_ext = os.path.splitext(file)
        unique_file = filename_ext[0]+"_unique"+filename_ext[1]
        with open(unique_file, 'w') as f:
          for line in lines:
            f.write(str(int(line.split()[0])+max_node_id)+" "+str(int(line.split()[1])+max_node_id)+"\n")
        file = unique_file


      g = nx.read_edgelist(file, nodetype=int, create_using=nx.DiGraph)
      G = nx.compose(G, g)
      G = nx.convert_node_labels_to_integers(G, first_label=1, label_attribute='old_label')
      max_node_id = max(list(G.nodes)) + 1

    all_shortest_paths = nx.shortest_path_length(G)
    for all_paths in all_shortest_paths:
        src = all_paths[0]
        for dest in all_paths[1]:
          dataset.append([src, dest, all_paths[1][dest]])
    
    # nx.draw(G)    
    return dataset #, trainset

def generate_graph_trainset(graphs, train_pct):
  # TODO: change to random computed distances rather than random landmarks
  trainset = []
  for gr in nx.weakly_connected_components(graphs):
    graph = graphs.subgraph(gr)
    num_nodes = round(graph.number_of_nodes() * train_pct)
    random_nodes = sample(list(graph.nodes()), num_nodes)
    for landmark in random_nodes:
      shortest_paths = nx.shortest_path_length(graph, source=landmark, weight='weight')
      for dest in shortest_paths:
        trainset.append([landmark, dest, shortest_paths[dest]])

  return trainset

def get_edge_list(filename):
    data = []
    with open(filename) as f:
        data = [ list(map(int, line.split()+[1])) for line in f.readlines()]

    return data


def generate_cycles_dataset(lg_N):
  N = 2**lg_N  # N: Max number of nodes in a cycle

  dataset = []
  node_id = {}
  counter = 0
  for n in range(3, N):  # For each cycle
    for i in range(n):  # For each node in each cycle
      node_id[(n, i)] = counter
      counter += 1

  for n in range(3, N):  # For each cycle
    for i in range(n):  # For each node in each cycle
      for j in range(n):
        if abs(i-j) <= 4:
          for k in range(5):
            dataset.append([node_id[(n, i)], node_id[(n, j)],
                           min(abs(i-j), n-abs(i-j))])
        dataset.append([node_id[(n, i)], node_id[(n, j)],
                       min(abs(i-j), n-abs(i-j))])

  return dataset


def save_dataset_to_file(dataset, filename):
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  with open(filename, 'w') as f:
    for data in dataset:
      f.write(" ".join(str(x) for x in data) + "\n")
  return True

def load_dataset_from_file(filename):
  with open(filename) as f:
    data = [ list(map(int, line.split())) for line in f.readlines() ]
  return data

