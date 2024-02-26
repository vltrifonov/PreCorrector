import jax.numpy as jnp

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
    
def graph_to_low_tri_mat(graph):
    L = jnp.zeros([graph.n_node.item()]*2)
    L = L.at[graph.senders, graph.receivers].set(graph.edges)
    return jnp.tril(L)