import random
import networkx as nx


def generate_random_tree(n):
  parent = {}
  nodes = list(range(n))
  random.shuffle(nodes)
  root = nodes[0]
  parent[root] = root
  for i in range(1, len(nodes)):
    parent[nodes[i]] = nodes[random.randrange(i)]
  return parent


def get_node_depths(tree):
    depths = {}  # size of tree, node:level
    #curr = root
    for node in tree:
        level = 0
        curr = node
        while tree[curr] != curr:
            if curr in depths:
                level += depths[curr]
                break  
            curr = tree[curr]
            level += 1
        depths[node] = level
    return depths

def get_distance(tree, src, dest, depths):
    s = src; d = dest
    while s != d:
        if depths[d] > depths[s]:
            d = tree[d]
        elif depths[s] > depths[d]:
            s = tree[s]
        else:
            if s != d and depths[s] == depths[d]:
                d = tree[d]
    lca = s
    src_depth = depths[src] - depths[lca]
    dest_depth = depths[dest] - depths[lca]
    return (src_depth + dest_depth)


# src, dst, distance_label
#
def generate_dataset(N):
    dataset = []     # node1, node2, distance
    node_id = {}
    id_to_node = {}
    node_pairs = {}  # (node1, node2): distance
    
    counter = 0
    trees = {}
    depths_graph = {}
    # give each node a unique id
    for n in range(3, N): # For each tree
        trees[n] = generate_random_tree(n)
        tree = trees[n]
        depths_graph[n] = get_node_depths(tree)        
        for node in tree: # For each node in the tree
            node_id[(n, node)] = counter
            id_to_node[counter] = n, node
            counter += 1
    
    for n in range(3, N):
        tree = trees[n]
        depths = depths_graph[n]
        for node in tree:
            for other_node in tree:
                dist = get_distance(tree, node, other_node, depths)
                dataset.append([node_id[(n, node)], node_id[(n, other_node)], dist])
                node_pairs[(node_id[(n, node)], node_id[(n, other_node)])] = dist
                
    return dataset
# generate_dataset(10)

# get edge adjacency list from files
# create the graph
# for each node in graph, get the distance to every other node
# 

def get_graph_from_files(fileslist):
    dataset = []
    
    for file in fileslist:
        # create graph with networkx
        g = nx.read_edgelist(file, nodetype=int, create_using=nx.DiGraph)
        all_shortest_paths = nx.shortest_path_length(g)
        for all_paths in all_shortest_paths:
            src = all_paths[0]
            for dest in all_paths[1]:
              dataset.append([src, dest, all_paths[1][dest]])

        # optimize loops to use just one networkx function
        nx.draw(g)
    return dataset


a = get_graph_from_files(["DistanceLabelling/datasets/ENZYMES_g1/ENZYMES_g1.edges"])
print(a)