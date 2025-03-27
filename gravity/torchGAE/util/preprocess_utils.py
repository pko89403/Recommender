import os
from random import sample 

import scipy.sparse as sp
import numpy as np
import pandas as pd
import igraph as ig
import torch

from util.deco_util import timeit


@timeit
def load_data(path, dataset):
    df = None

    if dataset == 'cora':
        cora_df = pd.read_csv(
            os.path.join(path, "cora.cites"),
            delimiter="\t",
            header=None,
            dtype=int,
        )
        df = cora_df[[0, 1]]
        
    elif dataset == 'artists':
        artist_df = pd.read_csv(
            os.path.join(path, "artists.csv"),
            sep=","
        )
        df = artist_df
    
    elif dataset == 'google':
        google_df = pd.read_csv(
            os.path.join(path, "GoogleNw.txt"),
            delimiter="	",
            header=None, 
            dtype=str
        )
        df = google_df
        

    directed_graph = ig.Graph.DataFrame(df, directed=True)
    
    """
        GET VERTEX NAME FROM INDEX
        vertex_names = directed_graph.vs[0]
        
        GET VERTEX INDEX FROM NAME 
        vertex_name_from_id = directed_graph.vs.find(name="1000012")
        
        PRINT
        print(vertex_names, vertex_name_from_id)
    """

    index2node = dict()
    node2index = dict()
    for idx, vertex in enumerate(directed_graph.vs()):

        index2node[str(idx)] = str(vertex["name"])
        node2index[str(vertex["name"])] = str(idx)
            
    """
        GET NODES RELATED TO SPECIFIC NODE
        neighbors = directed_graph.neighbors("1000012", mode="in/out")
        for neighbor in neighbors:
            print(directed_graph.vs[neighbor])
        
    """    
    
    # node id(index)를 기준으로 adjacency_matrix를 생성한다.
    try:
        # adj = directed_graph.get_adjacency_sparse(attribute='weight')
        adj = directed_graph.get_adjacency_sparse()
    except Exception as e:
        print(f"Exception Warining - {e}")
        adj = directed_graph.get_adjacency_sparse()
        

    """
       우 상 삼각 행렬이 OUT 방향이다. ( 1, 1254), (1, 1854), (1, 2399)
        ___
        \  |
         \ |
          \| 
       좌 하 삼각 행렬이 IN 방향이다. (? , 1)
    """

    # sp.identity : identity matrix in sparse format 
    # [[1 0]
    #  [0 1]]
    features = sp.identity(adj.shape[0]).tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((features.row, features.col)).astype(np.int))
    values = torch.from_numpy(features.data)
    shape = torch.Size(features.shape)

    features = torch.sparse.FloatTensor(indices, values, shape)

    return adj.T, features, index2node

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo() # return COOrdinate 표현 ( row, col, data )
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose() 
    # row = [0, 0, 0, 1, 2] col = [0, 1, 2, 2, 3]
    # -> coords = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 3]]
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

@timeit
def preprocess_graph(adj):
    """preprocess_graph
        output dereee normalization

    Args:
        adj (_type_): _description_

    Returns:
        _type_: _description_
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0]) # self-loop

    rowsum = np.array(adj_.sum(1))

    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int_))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

@timeit
def get_distribution(
    measure="uniform",
    alpha=1.0,
    adj=None):

    if measure == "uniform":
        proba = np.ones(adj.shape[0])

    elif measure == "degree":
        # 하나의 vertex가 있을 때, vertex에 연결된 edge의 갯수

        G = ig.Graph.Weighted_Adjacency(matrix=adj, loops=False)        
        degrees = G.degree(G.vs, mode="all")
        proba = np.power(degrees, alpha)

    elif measure == "core":
        # k-core 이론을 기준으로 노드 들을 그룹화하는 함수 
        # 노드들을 그룹으로 묶었을 때, 최소 k개 만큼 다른 노드와 연결관계를 가지는 경우
        G = ig.Graph.Weighted_Adjacency(matrix=adj, loops=False)
        coreness = G.coreness(mode="all")
        proba = np.power(coreness, alpha)

    else:
        raise ValueError("Undefined Sampling Method!")

    # Normalization
    proba = proba / np.sum(proba)

    return proba
  
def mask_test_edges_org(adj, test_percent=10., val_percent=5.):
    """ Randomly removes some edges from original graph to create
    test and validation sets for link prediction task
    :param adj: complete sparse adjacency matrix of the graph
    :param test_percent: percentage of edges in test set
    :param val_percent: percentage of edges in validation set
    :return: train incomplete adjacency matrix, validation and test sets
    """
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[None, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    edges_positive, _, _ = sparse_to_tuple(adj)
    # Filtering out edges from lower triangle of adjacency matrix
    edges_positive = edges_positive[edges_positive[:,1] > edges_positive[:,0],:]

    # number of positive (and negative) edges in test and val sets:
    num_test = int(np.floor(edges_positive.shape[0] / (100. / test_percent)))
    num_val = int(np.floor(edges_positive.shape[0] / (100. / val_percent)))

    # sample positive edges for test and val sets:
    edges_positive_idx = np.arange(edges_positive.shape[0])
    np.random.shuffle(edges_positive_idx)
    val_edge_idx = edges_positive_idx[:num_val]
    test_edge_idx = edges_positive_idx[num_val:(num_val + num_test)]
    test_edges = edges_positive[test_edge_idx] # positive test edges
    val_edges = edges_positive[val_edge_idx] # positive val edges
    train_edges = np.delete(edges_positive, np.hstack([test_edge_idx, val_edge_idx]), axis = 0) # positive train edges

    # the above strategy for sampling without replacement will not work for
    # sampling negative edges on large graphs, because the pool of negative
    # edges is much much larger due to sparsity, therefore we'll use
    # the following strategy:
    # 1. sample random linear indices from adjacency matrix WITH REPLACEMENT
    # (without replacement is super slow). sample more than we need so we'll
    # probably have enough after all the filtering steps.
    # 2. remove any edges that have already been added to the other edge lists
    # 3. convert to (i,j) coordinates
    # 4. swap i and j where i > j, to ensure they're upper triangle elements
    # 5. remove any duplicate elements if there are any
    # 6. remove any diagonal elements
    # 7. if we don't have enough edges, repeat this process until we get enough
    positive_idx, _, _ = sparse_to_tuple(adj) # [i,j] coord pairs for all true edges
    positive_idx = positive_idx[:,0]*adj.shape[0] + positive_idx[:,1] # linear indices
    test_edges_false = np.empty((0,2),dtype='int64')
    idx_test_edges_false = np.empty((0,),dtype='int64')

    while len(test_edges_false) < len(test_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_test - len(test_edges_false)), replace = True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique = True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx,colidx)).transpose()
        # step 4:
        lowertrimask = coords[:,0] > coords[:,1]
        coords[lowertrimask] = coords[lowertrimask][:,::-1]
        # step 5:
        coords = np.unique(coords, axis = 0) # note: coords are now sorted lexicographically
        np.random.shuffle(coords) # not anymore
        # step 6:
        coords = coords[coords[:,0] != coords[:,1]]
        # step 7:
        coords = coords[:min(num_test, len(idx))]
        test_edges_false = np.append(test_edges_false, coords, axis = 0)
        idx = idx[:min(num_test, len(idx))]
        idx_test_edges_false = np.append(idx_test_edges_false, idx)

    val_edges_false = np.empty((0,2), dtype = 'int64')
    idx_val_edges_false = np.empty((0,), dtype = 'int64')
    while len(val_edges_false) < len(val_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_val - len(val_edges_false)), replace = True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_val_edges_false, assume_unique = True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx,colidx)).transpose()
        # step 4:
        lowertrimask = coords[:,0] > coords[:,1]
        coords[lowertrimask] = coords[lowertrimask][:,::-1]
        # step 5:
        coords = np.unique(coords, axis = 0) # note: coords are now sorted lexicographically
        np.random.shuffle(coords) # not any more
        # step 6:
        coords = coords[coords[:,0] != coords[:,1]]
        # step 7:
        coords = coords[:min(num_val, len(idx))]
        val_edges_false = np.append(val_edges_false, coords, axis = 0)
        idx = idx[:min(num_val, len(idx))]
        idx_val_edges_false = np.append(idx_val_edges_false, idx)

    # Re-build adj matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    return adj_train, val_edges, val_edges_false, test_edges, test_edges_false


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = np.arange(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false