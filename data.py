import jax.numpy as jnp
import jraph

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

def has_edge(graph, node1, node2):
    node1_connected_to = graph.receivers[graph.senders == node1]
    connect = node2 in node1_connected_to
    return connect

def is_bi_direc_edge(graph, node1, node2):
    n1_to_n2 = has_edge(graph, node1, node2)
    n2_to_n1 = has_edge(graph, node2, node1)
    return n1_to_n2 and n2_to_n1
    
def bi_direc_indx(graph, node1, node2):
    bi_direc_pairs = []
    for n1 in jnp.arange(graph.n_node):
        for n2 in jnp.arange(n1+1, graph.n_node):
            if is_bi_direc_edge(graph, node1, node2):
                indx1, indx2 = 
                bi_direc_pairs.append([indx1, indx2])
        