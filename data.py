import jax.numpy as jnp
import jraph

from utils import has_edge, is_bi_direc_edge, edge_index

def direc_graph_from_linear_system(A, b):
    '''Matrix `A` should be sparse.'''
    node_features = jnp.asarray(b)
    senders, receivers = jnp.nonzero(A)
    edge_features = A[senders, receivers]
    n_node = jnp.array([len(node_features)])
    n_edge = jnp.array([len(senders)])
    graph = jraph.GraphsTuple(nodes=node_features, edges=edge_features, senders=senders,
                              receivers=receivers, n_node=n_node, n_edge=n_edge, globals=None)
    return graph 
    
def bi_direc_indx(graph):
    '''Returns indices of edges which corresponds to bi-direcional connetions.'''
    bi_edge_pairs = []
    for n1 in jnp.arange(graph.n_node.item()):
        for n2 in jnp.arange(n1+1, graph.n_node.item()):
            if is_bi_direc_edge(graph, n1, n2):
                indx1 = edge_index(graph, n1, n2)
                indx2 = edge_index(graph, n2, n1)
                bi_edge_pairs.append([indx1, indx2])
    bi_edge_pairs = jnp.stack(jnp.asarray(bi_edge_pairs))
    return bi_edge_pairs

def bi_direc_edge_avg(graph, indices):
    edges_new = graph.edges.copy()
    for i in range(len(indices)):
        edges_new = edges_new.at[indices[i]].set(jnp.mean(edges_new[indices[i]]))    
    graph_new = jraph.GraphsTuple(nodes=graph.nodes, edges=edges_new, senders=graph.senders,
                                  receivers=graph.receivers, n_node=graph.n_node, n_edge=graph.n_edge,
                                  globals=None)
    return graph_new