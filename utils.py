import jax.numpy as jnp

def bi_direc_edge_avg():
    
    pass
    
def graph_to_low_tri_mat(graph):
    L = jnp.zeros([len(graph.node_features), len(graph.node_features)])
    L = L.at[graph.senders, graph.receivers].set(graph.edges)
    return jnp.tril(L)
    