import itertools
from multiprocessing import Pool 
from multiprocessing import cpu_count
from functools import partial

import numpy as np

from util.deco_util import timeit

@timeit
def parallel_node_sampling(total_epochs, n_cores, adj, dist, num_samples, replace):
    chunks = np.array_split([idx for idx in range(total_epochs)], n_cores)
    
    if cpu_count() < n_cores:
        raise ValueError("The number of CPU's specified exceed the amount available")
    
    func=node_sampling_merge
    func_params = {
        "adj": adj,
        "dist": dist,
        "num_samples": num_samples,
        "replace": replace
    }
        
    pool = Pool(n_cores)
    res = pool.map(
        partial(
            func,
            params=func_params
        ),
        chunks
    )
    pool.close()
    pool.join()
    
    return list(itertools.chain.from_iterable(res))

def node_sampling_chunk(adj, distribution, num_samples, replace=True):
    sampled_nodes = np.random.choice(
        adj.shape[0],
        size=num_samples,
        replace=replace,
        p=distribution
    )
    return sampled_nodes

def node_sampling_merge(chunks, params):
    adj = params["adj"]
    distribution = params["dist"]
    num_samples = params["num_samples"]
    replace = params["replace"]
    
    results = []
    while True:
        sampled_nodes = node_sampling_chunk(
            adj=adj,
            distribution=distribution,
            num_samples=num_samples,
            replace=replace
        )
        
        if adj[sampled_nodes,:][:,sampled_nodes].sum() <= 0.0:
            continue
            
        results.append(sampled_nodes)
        if len(results) > len(chunks):
            break

    return results