import jax.numpy as jnp
from jax.experimental import sparse as jsparse
import equinox as eqx

from utils import graph_to_low_tri_mat_sparse, graph_tril, graph_to_low_tri_mat
from data import bi_direc_edge_avg

class PrecNetLearnAll(eqx.Module):
    NodeEncoder: eqx.Module
    EdgeEncoder: eqx.Module
    MessagePass: eqx.Module
    EdgeDecoder: eqx.Module

    def __init__(self, NodeEncoder, EdgeEncoder, MessagePass, EdgeDecoder):
        super(PrecNetLearnAll, self).__init__()
        self.NodeEncoder = NodeEncoder
        self.EdgeEncoder = EdgeEncoder
        self.MessagePass = MessagePass
        self.EdgeDecoder = EdgeDecoder
        return
    
    def __call__(self, nodes, edges, receivers, senders, bi_edges_indx):
        nodes = self.NodeEncoder(nodes[None, ...])
        edges = self.EdgeEncoder(edges[None, ...])
        nodes, edges, receivers, senders = self.MessagePass(nodes, edges, receivers, senders)
        edges = bi_direc_edge_avg(edges, bi_edges_indx)
        edges = self.EdgeDecoder(edges)
        nodes, edges, receivers, senders = graph_tril(nodes, jnp.squeeze(edges), receivers, senders)
        low_tri = graph_to_low_tri_mat_sparse(nodes, edges, receivers, senders)
        return low_tri

    
class PrecNetCopyDiag(eqx.Module):
    NodeEncoder: eqx.Module
    EdgeEncoder: eqx.Module
    MessagePass: eqx.Module
    EdgeDecoder: eqx.Module

    def __init__(self, NodeEncoder, EdgeEncoder, MessagePass, EdgeDecoder):
        super(PrecNetCopyDiag, self).__init__()
        self.NodeEncoder = NodeEncoder
        self.EdgeEncoder = EdgeEncoder
        self.MessagePass = MessagePass
        self.EdgeDecoder = EdgeDecoder
        return
    
    def __call__(self, nodes, edges, receivers, senders, bi_edges_indx):
         # Save main diagonal
        diag_edge_indx = jnp.diff(jnp.hstack([senders[:, None], receivers[:, None]]))
        diag_edge_indx = jnp.where(diag_edge_indx == 0, 1, 0)
        diag_edge_indx = jnp.nonzero(diag_edge_indx, size=nodes.shape[0], fill_value=jnp.nan)[0].astype(jnp.int32)
        diag_edge = edges.at[diag_edge_indx].get()
        
        nodes = self.NodeEncoder(nodes[None, ...])
        edges = self.EdgeEncoder(edges[None, ...])
        nodes, edges, receivers, senders = self.MessagePass(nodes, edges, receivers, senders)
        edges = bi_direc_edge_avg(edges, bi_edges_indx)
        edges = self.EdgeDecoder(edges)
        edges = edges.at[:, diag_edge_indx].set(diag_edge)
        
        nodes, edges, receivers, senders = graph_tril(nodes, jnp.squeeze(edges), receivers, senders)
        low_tri = graph_to_low_tri_mat_sparse(nodes, edges, receivers, senders)
        return low_tri
    
    
class PrecNetCopyDiagSqrt(eqx.Module):
    NodeEncoder: eqx.Module
    EdgeEncoder: eqx.Module
    MessagePass: eqx.Module
    EdgeDecoder: eqx.Module

    def __init__(self, NodeEncoder, EdgeEncoder, MessagePass, EdgeDecoder):
        super(PrecNetCopyDiagSqrt, self).__init__()
        self.NodeEncoder = NodeEncoder
        self.EdgeEncoder = EdgeEncoder
        self.MessagePass = MessagePass
        self.EdgeDecoder = EdgeDecoder
        return
    
    def __call__(self, nodes, edges, receivers, senders, bi_edges_indx):
         # Save main diagonal
        diag_edge_indx = jnp.diff(jnp.hstack([senders[:, None], receivers[:, None]]))
        diag_edge_indx = jnp.where(diag_edge_indx == 0, 1, 0)
        diag_edge_indx = jnp.nonzero(diag_edge_indx, size=nodes.shape[0], fill_value=jnp.nan)[0].astype(jnp.int32)
        diag_edge = edges.at[diag_edge_indx].get()
        
        nodes = self.NodeEncoder(nodes[None, ...])
        edges = self.EdgeEncoder(edges[None, ...])
        nodes, edges, receivers, senders = self.MessagePass(nodes, edges, receivers, senders)
        edges = bi_direc_edge_avg(edges, bi_edges_indx)
        edges = self.EdgeDecoder(edges)
        edges = edges.at[:, diag_edge_indx].set(jnp.sqrt(diag_edge))
        
        nodes, edges, receivers, senders = graph_tril(nodes, jnp.squeeze(edges), receivers, senders)
        low_tri = graph_to_low_tri_mat_sparse(nodes, edges, receivers, senders)
        return low_tri

    
class PrecNetRigidLDLT(eqx.Module):
    NodeEncoder: eqx.Module
    EdgeEncoder: eqx.Module
    MessagePass: eqx.Module
    EdgeDecoder: eqx.Module

    def __init__(self, NodeEncoder, EdgeEncoder, MessagePass, EdgeDecoder):
        super(PrecNetRigidLDLT, self).__init__()
        self.NodeEncoder = NodeEncoder
        self.EdgeEncoder = EdgeEncoder
        self.MessagePass = MessagePass
        self.EdgeDecoder = EdgeDecoder
        return
    
    def __call__(self, nodes, edges, receivers, senders, bi_edges_indx):
         # Save main diagonal
        diag_edge_indx = jnp.diff(jnp.hstack([senders[:, None], receivers[:, None]]))
        diag_edge_indx = jnp.where(diag_edge_indx == 0, 1, 0)
        diag_edge_indx = jnp.nonzero(diag_edge_indx, size=nodes.shape[0], fill_value=jnp.nan)[0].astype(jnp.int32)
        diag_edge = edges.at[diag_edge_indx].get()
        D = jsparse.eye(diag_edge.shape[0]) * diag_edge
        
        nodes = self.NodeEncoder(nodes[None, ...])
        edges = self.EdgeEncoder(edges[None, ...])
        nodes, edges, receivers, senders = self.MessagePass(nodes, edges, receivers, senders)
        edges = bi_direc_edge_avg(edges, bi_edges_indx)
        edges = self.EdgeDecoder(edges)
        
        edges = edges.at[:, diag_edge_indx].set(1)
        
        nodes, edges, receivers, senders = graph_tril(nodes, jnp.squeeze(edges), receivers, senders)
        low_tri = graph_to_low_tri_mat_sparse(nodes, edges, receivers, senders)
        return low_tri, D


class PrecNetLDLT(eqx.Module):
    #TODO
    NodeEncoder: eqx.Module
    EdgeEncoder: eqx.Module
    MessagePass: eqx.Module
    EdgeDecoder: eqx.Module

    def __init__(self, NodeEncoder, EdgeEncoder, MessagePass, EdgeDecoder):
        super(PrecNetLDLT, self).__init__()
        self.NodeEncoder = NodeEncoder
        self.EdgeEncoder = EdgeEncoder
        self.MessagePass = MessagePass
        self.EdgeDecoder = EdgeDecoder
        return
    
    def __call__(self, nodes, edges, receivers, senders, bi_edges_indx):
         # Save main diagonal
        diag_edge_indx = jnp.diff(jnp.hstack([senders[:, None], receivers[:, None]]))
        diag_edge_indx = jnp.where(diag_edge_indx == 0, 1, 0)
        diag_edge_indx = jnp.nonzero(diag_edge_indx, size=nodes.shape[0], fill_value=jnp.nan)[0].astype(jnp.int32)
        diag_edge = edges.at[diag_edge_indx].get()
        D = jsparse.eye(diag_edge.shape[0]) * diag_edge
        
        nodes = self.NodeEncoder(nodes[None, ...])
        edges = self.EdgeEncoder(edges[None, ...])
        nodes, edges, receivers, senders = self.MessagePass(nodes, edges, receivers, senders)
        edges = bi_direc_edge_avg(edges, bi_edges_indx)
        edges = self.EdgeDecoder(edges)
        nodes, edges, receivers, senders = graph_tril(nodes, jnp.squeeze(edges), receivers, senders)
        low_tri = graph_to_low_tri_mat_sparse(nodes, edges, receivers, senders)
  
         # Normalize L
#         diag_edge_to_norm = edges.at[diag_edge_indx].get()
#         S_inv = jsparse.eye(diag_edge_to_norm.shape[0]) * (1 / diag_edge_to_norm)
#         low_tri = jsparse.sparsify(lambda A, B: A @ B)(low_tri, S_inv)
        return low_tri, D, S_inv