from typing import Callable

from jax import random, vmap
import jax.numpy as jnp
import jax.tree_util as tree
import jax.nn as jnn
from jax.experimental import sparse as jsparse

from jax.ops import segment_sum
import equinox as eqx

class FullyConnectedNet(eqx.Module):
    layers: list
    act: Callable = eqx.field(static=True)
        
    def __init__(self, features, N_layers, key, act=jnn.relu):
        super(FullyConnectedNet, self).__init__()
        N_in, N_pr, N_out = features
        keys = random.split(key, N_layers)
        Ns = [N_in,] + [N_pr,] * (N_layers - 1) + [N_out,]
        self.layers = [eqx.nn.Conv1d(in_channels=N_in, out_channels=N_out, kernel_size=1, key=key) for N_in, N_out, key in zip(Ns[:-1], Ns[1:], keys)]
        self.act = act
        return
    
    def __call__(self, x):
        for l in self.layers[:-1]:
            x = l(x)
            x = self.act(x)
        x = self.layers[-1](x)
        return x
    
    
class MessagePassingNoDiag(eqx.Module):
    update_edge_fn: eqx.Module
    update_node_fn: eqx.Module
    aggregate_edges_for_nodes_fn: Callable = eqx.field(static=True)
    mp_rounds: int = eqx.field(static=True)
        
    def __init__(self, update_edge_fn, update_node_fn, mp_rounds, aggregate_edges_for_nodes_fn=segment_sum):
        super(MessagePassingNoDiag, self).__init__()
        self.update_edge_fn = update_edge_fn
        self.update_node_fn = update_node_fn
        self.aggregate_edges_for_nodes_fn = aggregate_edges_for_nodes_fn
        self.mp_rounds = mp_rounds
        return        
        
    def __call__(self, nodes, edges, receivers, senders):
        for _ in range(self.mp_rounds):
            nodes = self._update_nodes(nodes, edges, receivers, senders)
            edges = self._update_edges(nodes, edges, receivers, senders)
        return nodes, edges, receivers, senders
    
    def _update_nodes(self, nodes, edges, receivers, senders):
        sum_n_node = tree.tree_leaves(nodes)[0].shape[1]
        sent_attributes = vmap(
            tree.tree_map, 
            in_axes=(None, 0), out_axes=(0)
        )(lambda e: self.aggregate_edges_for_nodes_fn(e, senders, sum_n_node), edges)
    
        received_attributes = vmap(
            tree.tree_map, 
            in_axes=(None, 0), out_axes=(0)
        )(lambda e: self.aggregate_edges_for_nodes_fn(e, receivers, sum_n_node), edges)
        
        nodes = self.update_node_fn(jnp.concatenate([nodes, sent_attributes, received_attributes], axis=0))
        return nodes
    
    def _update_edges(self, nodes, edges, receivers, senders):
        sent_attributes = tree.tree_map(lambda n: n[:, senders], nodes)
        received_attributes = tree.tree_map(lambda n: n[:, receivers], nodes)

#         # Find non-main-diagonal edges
        non_diag_edge_idx = jnp.diff(jnp.hstack([senders[:, None], receivers[:, None]]))
        non_diag_edge_idx = jnp.nonzero(non_diag_edge_idx, size=senders.shape[0]-nodes.shape[1], fill_value=jnp.nan)[0].astype(jnp.int32)
        
        feat_in = jnp.concatenate([
            edges[:, non_diag_edge_idx],
            sent_attributes[:, non_diag_edge_idx],
            received_attributes[:, non_diag_edge_idx]
        ], axis=0)
        edges_upd = self.update_edge_fn(feat_in)
        edges = edges.at[:, non_diag_edge_idx].set(edges_upd)
        return edges

    
class MessagePassingWithDiag(eqx.Module):
    update_edge_fn: eqx.Module
    update_node_fn: eqx.Module
    aggregate_edges_for_nodes_fn: Callable = eqx.field(static=True)
    mp_rounds: int = eqx.field(static=True)
        
    def __init__(self, update_edge_fn, update_node_fn, mp_rounds, aggregate_edges_for_nodes_fn=segment_sum):
        super(MessagePassingWithDiag, self).__init__()
        self.update_edge_fn = update_edge_fn
        self.update_node_fn = update_node_fn
        self.aggregate_edges_for_nodes_fn = aggregate_edges_for_nodes_fn
        self.mp_rounds = mp_rounds
        return        
        
    def __call__(self, nodes, edges, receivers, senders):
        for _ in range(self.mp_rounds):
            nodes = self._update_nodes(nodes, edges, receivers, senders)
            edges = self._update_edges(nodes, edges, receivers, senders)
        return nodes, edges, receivers, senders
    
    def _update_nodes(self, nodes, edges, receivers, senders):
        sum_n_node = tree.tree_leaves(nodes)[0].shape[1]
        sent_attributes = vmap(
            tree.tree_map, 
            in_axes=(None, 0), out_axes=(0)
        )(lambda e: self.aggregate_edges_for_nodes_fn(e, senders, sum_n_node), edges)
    
        received_attributes = vmap(
            tree.tree_map, 
            in_axes=(None, 0), out_axes=(0)
        )(lambda e: self.aggregate_edges_for_nodes_fn(e, receivers, sum_n_node), edges)
        
        nodes = self.update_node_fn(jnp.concatenate([nodes, sent_attributes, received_attributes], axis=0))
        return nodes
    
    def _update_edges(self, nodes, edges, receivers, senders):
        sent_attributes = tree.tree_map(lambda n: n[:, senders], nodes)
        received_attributes = tree.tree_map(lambda n: n[:, receivers], nodes)
        edges = self.update_edge_fn(jnp.concatenate([edges, sent_attributes, received_attributes], axis=0))
        return edges