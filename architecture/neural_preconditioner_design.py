import jax
import jax.numpy as jnp
import equinox as eqx

from data.graph_utils import bi_direc_edge_avg, graph_to_spmatrix, symm_graph_tril

class PreCorrector(eqx.Module):
    '''L = L + alpha * GNN(L)'''
    EdgeEncoder: eqx.Module
    MessagePass: eqx.Module
    EdgeDecoder: eqx.Module
    alpha: jax.Array

    def __init__(self, EdgeEncoder, MessagePass, EdgeDecoder, alpha):
        super(PreCorrector, self).__init__()
        self.EdgeEncoder = EdgeEncoder
        self.MessagePass = MessagePass
        self.EdgeDecoder = EdgeDecoder
        self.alpha = alpha
        return
    
    def __call__(self, train_graph):
        nodes, edges_init, senders, receivers = train_graph
        norm = jnp.abs(edges_init).max()
        edges = edges_init / norm
        
        edges = self.EdgeEncoder(edges[None, ...])
        nodes, edges, senders, receivers = self.MessagePass(nodes, edges, senders, receivers)
        edges = self.EdgeDecoder(edges)[0, ...]
        edges = edges_init + self.alpha * (edges * norm)
        
        nodes, edges, senders, receivers = symm_graph_tril(nodes, jnp.squeeze(edges), senders, receivers)
        low_tri = graph_to_spmatrix(nodes, edges, senders, receivers)
        return low_tri
    
class NaiveGNN(eqx.Module):
    '''GNN operates on A.
    Perseving diagonal as: diag(A) = diag(D) from A = LDL^T'''
    NodeEncoder: eqx.Module
    EdgeEncoder: eqx.Module
    MessagePass: eqx.Module
    EdgeDecoder: eqx.Module

    def __init__(self, NodeEncoder, EdgeEncoder, MessagePass, EdgeDecoder):
        super(NaiveGNN, self).__init__()
        self.NodeEncoder = NodeEncoder
        self.EdgeEncoder = EdgeEncoder
        self.MessagePass = MessagePass
        self.EdgeDecoder = EdgeDecoder
        return    
    
    def __call__(self, train_graph, bi_edges_indx, lhs_graph):
        lhs_nodes, lhs_edges, lhs_senders, lhs_receivers = lhs_graph
        nodes, edges, senders, receivers = train_graph
        
         # Save main diagonal in initial lhs from PDE
        diag_edge_indx_lhs = jnp.diff(jnp.hstack([lhs_senders[:, None], lhs_receivers[:, None]]))
        diag_edge_indx_lhs = jnp.argwhere(diag_edge_indx_lhs == 0, size=lhs_nodes.shape[0], fill_value=jnp.nan)[:, 0].astype(jnp.int32)
        diag_edge = lhs_edges.at[diag_edge_indx_lhs].get(mode='drop', fill_value=0)
        
        # Main diagonal in padded lhs for train
        diag_edge_indx = jnp.diff(jnp.hstack([senders[:, None], receivers[:, None]]))
        diag_edge_indx = jnp.argwhere(diag_edge_indx == 0, size=nodes.shape[0], fill_value=jnp.nan)[:, 0].astype(jnp.int32)
                
        nodes = self.NodeEncoder(nodes[None, ...])
        edges = self.EdgeEncoder(edges[None, ...])
        nodes, edges, senders, receivers = self.MessagePass(nodes, edges, senders, receivers)
        edges = bi_direc_edge_avg(edges, bi_edges_indx)
        edges = self.EdgeDecoder(edges)[0, ...]
        
        # Put the real diagonal into trained lhs
        edges = edges.at[diag_edge_indx].set(jnp.sqrt(diag_edge), mode='drop')
        
        nodes, edges, senders, receivers = symm_graph_tril(nodes, jnp.squeeze(edges), senders, receivers)
        low_tri = graph_to_spmatrix(nodes, edges, senders, receivers)
        return low_tri