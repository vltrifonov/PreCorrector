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
    
def edge_index(graph, node1, node2):
    send_indx = jnp.nonzero(graph.senders == node1)[0][0]         # First edge index of this sender
    node1_connected_to = graph.receivers[graph.senders == node1]  # To what nodes first node is connected
    rec_indx = jnp.nonzero(node1_connected_to == node2)[0][0]     # Index of needed node within 
    return send_indx + rec_indx
    
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