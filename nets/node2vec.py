# import pandas as pd
# import torch
# from itertools import permutations
# import numpy as np
# from fastnode2vec import Graph, Node2Vec
# from stellargraph.data import BiasedRandomWalk
# from stellargraph import StellarGraph
#
# # from gensim.models import Word2Vec
#
#
# class Node2VecWrapper:
#     @staticmethod
#     def calc(input_matrix):
#         EDM = input_matrix.numpy()
#         iEDM = np.reciprocal(EDM)
#         squared_iEDM = np.square(iEDM)
#
#         # matrix = np.where(squared_iEDM == np.inf, 0, squared_iEDM)
#         # weight_list = matrix[~np.eye(matrix.shape[0],dtype=bool)].reshape(1,-1)[0].tolist()
#         #
#         # comb = np.array(list(permutations(range(len(matrix)), 2))).T
#         # source = list(comb[0])
#         # target = list(comb[1])
#         # square_numeric_edges = pd.DataFrame(
#         #     {"source": source, "target": target, "weight": weight_list}
#         # )
#
#         # stellarGraph = StellarGraph(edges=square_numeric_edges)
#         #
#         # rw = BiasedRandomWalk(stellarGraph)
#         #
#         # weighted_walks = rw.run(
#         #     nodes=stellarGraph.nodes(),  # root nodes
#         #     length=100,  # maximum length of a random walk
#         #     n=10,  # number of random walks per root node
#         #     p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
#         #     q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
#         #     weighted=True,  # for weighted random walks
#         #     seed=42,  # random seed fixed for reproducibility
#         # )
#         #
#         # weighted_model = Word2Vec(
#         #     weighted_walks, vector_size=128, window=5, sg=1, workers=10, epochs=1
#         # )
#
#         # graph = Graph([(int(x[0]), int(x[1]), x[2]) for x in square_numeric_edges.to_numpy()],
#         #               directed=True, weighted=True)
#         # weighted_model = Node2Vec(graph, dim=128, walk_length=1000, context=3, p=2.0, q=0.5, workers=10)
#         # weighted_model.train(epochs=100)
#         #
#         # node_ids = weighted_model.wv.index_to_key  # list of node IDs
#         # right_order = [node_ids.index(num) for num in range(len(node_ids))]
#         # right_ordered_model = torch.Tensor(weighted_model.wv.vectors[right_order])
#
#         return right_ordered_model[0], right_ordered_model[1:]
#
# def similarity(List1, List2):
#     from scipy import spatial
#     return  1 - spatial.distance.cosine(list(List1), list(List2))
