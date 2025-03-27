import igraph as ig 
import pandas as pd  

df = pd.DataFrame({'name1': ['$hort, Too', '$hort, Too'], 'name2': ['Alexander, Khandi', 'B-Real'], 'weight': [0.083333, 0.083333]}) 

# mygraph = ig.Graph.DataFrame(mydata) 

mygraph = ig.Graph(directed=True)
for i, row in df.iterrows():
    mygraph.add_vertex(row.name1)
    mygraph.add_vertex(row.name2)
    mygraph.add_edge(row.name1, row.name2, weight=row.weight) 

print(mygraph)

adj = mygraph.get_adjacency_sparse(attribute='weight')
print(adj.todense())