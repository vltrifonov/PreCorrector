import jax
import jax.numpy as jnp
import equinox as eqx

from data.graph_utils import bi_direc_edge_avg, graph_to_low_tri_mat_sparse, graph_tril

class PreCorrector(eqx.Module):
    '''L = L + alpha * GNN(L)'''
    NodeEncoder: eqx.Module
    EdgeEncoder: eqx.Module
    MessagePass: eqx.Module
    EdgeDecoder: eqx.Module
    alpha: jax.Array

    def __init__(self, NodeEncoder, EdgeEncoder, MessagePass, EdgeDecoder, alpha):
        super(CorrectionNet, self).__init__()
        self.NodeEncoder = NodeEncoder
        self.EdgeEncoder = EdgeEncoder
        self.MessagePass = MessagePass
        self.EdgeDecoder = EdgeDecoder
        self.alpha = alpha
        return
    
    def __call__(self, train_graph, bi_edges_indx):
        nodes, edges_init, receivers, senders = train_graph
        norm = jnp.abs(edges_init).max()
        edges = edges_init / norm
        
        nodes = self.NodeEncoder(nodes[None, ...])
        edges = self.EdgeEncoder(edges[None, ...])
        nodes, edges, receivers, senders = self.MessagePass(nodes, edges, receivers, senders)
        edges = bi_direc_edge_avg(edges, bi_edges_indx)
        edges = self.EdgeDecoder(edges)[0, ...]
        
        edges = edges * norm
        edges = edges_init + self.alpha * edges
        
        nodes, edges, receivers, senders = graph_tril(nodes, jnp.squeeze(edges), receivers, senders)
        low_tri = graph_to_low_tri_mat_sparse(nodes, edges, receivers, senders)
        return low_tri
    
class NaiveGNN(eqx.Module):
    '''GNN operates on A.
    Perseving diagonal as: diag(A) = diag(D) from A = LDL^T'''
    NodeEncoder: eqx.Module
    EdgeEncoder: eqx.Module
    MessagePass: eqx.Module
    EdgeDecoder: eqx.Module

    def __init__(self, NodeEncoder, EdgeEncoder, MessagePass, EdgeDecoder):
        super(PrecNet, self).__init__()
        self.NodeEncoder = NodeEncoder
        self.EdgeEncoder = EdgeEncoder
        self.MessagePass = MessagePass
        self.EdgeDecoder = EdgeDecoder
        return    
    
    def __call__(self, train_graph, bi_edges_indx, lhs_graph):
        lhs_nodes, lhs_edges, lhs_receivers, lhs_senders = lhs_graph
        nodes, edges, receivers, senders = train_graph
        
         # Save main diagonal in initial lhs from PDE
        diag_edge_indx_lhs = jnp.diff(jnp.hstack([lhs_senders[:, None], lhs_receivers[:, None]]))
        diag_edge_indx_lhs = jnp.argwhere(diag_edge_indx_lhs == 0, size=lhs_nodes.shape[0], fill_value=jnp.nan)[:, 0].astype(jnp.int32)
        diag_edge = lhs_edges.at[diag_edge_indx_lhs].get(mode='drop', fill_value=0)
        
        # Main diagonal in padded lhs for train
        diag_edge_indx = jnp.diff(jnp.hstack([senders[:, None], receivers[:, None]]))
        diag_edge_indx = jnp.argwhere(diag_edge_indx == 0, size=nodes.shape[0], fill_value=jnp.nan)[:, 0].astype(jnp.int32)
                
        nodes = self.NodeEncoder(nodes[None, ...])
        edges = self.EdgeEncoder(edges[None, ...])
        nodes, edges, receivers, senders = self.MessagePass(nodes, edges, receivers, senders)
        edges = bi_direc_edge_avg(edges, bi_edges_indx)
        edges = self.EdgeDecoder(edges)[0, ...]
        
        # Put the real diagonal into trained lhs
        edges = edges.at[diag_edge_indx].set(jnp.sqrt(diag_edge), mode='drop')
        
        nodes, edges, receivers, senders = graph_tril(nodes, jnp.squeeze(edges), receivers, senders)
        low_tri = graph_to_low_tri_mat_sparse(nodes, edges, receivers, senders)
        return low_tri