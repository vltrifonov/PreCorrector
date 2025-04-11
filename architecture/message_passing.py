from typing import Callable

import jax
from jax import vmap
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as tree

nodes_init_nodes_val = lambda nodes: nodes
nodes_init_ones = lambda nodes: jnp.ones_like(nodes).reshape([1, -1])

class MessagePassing_StaticDiag(eqx.Module):
    update_edge_fn: eqx.Module
    update_node_fn: eqx.Module
    aggregate_edges: Callable = eqx.field(static=True)
    mp_rounds: int = eqx.field(static=True)
    nodes_init_fn: Callable = eqx.field(static=True)
#     edge_norm: eqx.Module
#     node_norm: eqx.Module
        
    def __init__(self, update_edge_fn, update_node_fn, mp_rounds,
                 nodes_init_fn, aggregate_edges):
        super(MessagePassing_StaticDiag, self).__init__()
        self.update_edge_fn = update_edge_fn
        self.update_node_fn = update_node_fn
        self.aggregate_edges = aggregate_edges
        self.mp_rounds = mp_rounds
        self.nodes_init_fn = nodes_init_fn
#         self.edge_norm = eqx.nn.LayerNorm([16, 479200])
#         self.node_norm = eqx.nn.LayerNorm([1, 160000])
        return        
        
    def __call__(self, nodes, edges, senders, receivers):
#         print('\n\n\n!!! Forward start')
        nodes = self.nodes_init_fn(nodes)
#         print('  Input values')
#         print('   Nodes nan? ', jnp.any(jnp.isnan(nodes)))#jnp.min(jnp.abs(nodes)))#jnp.any(jnp.any(jnp.isnan(nodes))))
#         print('   Edges nan? ', jnp.any(jnp.isnan(edges)))#jnp.min(jnp.abs(edges)))#jnp.any(jnp.any(jnp.isnan(edges))))
            
        for i in range(self.mp_rounds):
#             print('Round', i)
            nodes = self._update_nodes(nodes, edges, senders, receivers)
#             nodes = self.node_norm(nodes)
#             nodes = vmap(self.node_norm, in_axes=(0), out_axes=(0))(nodes)
#             print('  After node update')
#             print('   NANs:', jnp.any(jnp.isnan(nodes)))
#             print('   MIN:', jnp.min(jnp.abs(nodes)))# jnp.min(jnp.abs(nodes)))#jnp.any(jnp.isnan(nodes)))
#             print('   MAX:', jnp.max(jnp.abs(nodes)))
#             print()
            edges = self._update_edges(nodes, edges, senders, receivers)
#             print('\n')
#             edges = self.edge_norm(edges)
#             edges = vmap(self.edge_norm, in_axes=(0), out_axes=(0))(edges)
#             print('  After edge update')
#             print('   MIN:', jnp.min(jnp.abs(edges)))# jnp.min(jnp.abs(nodes)))#jnp.any(jnp.isnan(nodes)))
#             print('   MAX:', jnp.max(jnp.abs(edges)))
#             print('   NANs:', jnp.any(jnp.isnan(edges)))
            
#             print('\n\n')
#         print('!!! Forward end')
        return nodes, edges, senders, receivers
    
    def _update_nodes(self, nodes, edges, senders, receivers):
        sum_n_node = tree.tree_leaves(nodes)[0].shape[-1]
        edges_by_receivers = edges * nodes[:, receivers] # Elemet-wise e_{i,j,t}v_{j,t}
#         edges_by_receivers = edges_by_receivers / jnp.abs(edges_by_receivers).max()
#         print('!!!!!!!!!!', edges_by_receivers.shape)
        sent_attributes = vmap(
            tree.tree_map,
            in_axes=(None, 0), out_axes=(0)
        )(lambda e: self.aggregate_edges(e, senders, sum_n_node), edges_by_receivers)
        
        nodes = self.update_node_fn(jnp.concatenate([nodes, sent_attributes], axis=0))
        return nodes
    
    def _update_edges(self, nodes, edges, senders, receivers):
        sent_attributes = tree.tree_map(lambda n: n[:, senders], nodes)
        received_attributes = tree.tree_map(lambda n: n[:, receivers], nodes)

        # Find non-main-diagonal edges
        non_diag_edge_idx = jnp.diff(jnp.hstack([senders[:, None], receivers[:, None]]))
        non_diag_edge_idx = jnp.nonzero(non_diag_edge_idx, size=senders.shape[0]-nodes.shape[1], fill_value=jnp.nan)[0].astype(jnp.int32)
        
        feat_in = jnp.concatenate([
            edges[:, non_diag_edge_idx],
            sent_attributes[:, non_diag_edge_idx],
            received_attributes[:, non_diag_edge_idx]
        ], axis=0)
        edges_upd = self.update_edge_fn(feat_in)
        edges = edges.at[:, non_diag_edge_idx].set(edges_upd, mode='drop')
        return edges


class MessagePassing_NotStaticDiag(MessagePassing_StaticDiag):
    def __init__(self, update_edge_fn, update_node_fn, mp_rounds,
                 nodes_init_fn, aggregate_edges):
        super().__init__(update_edge_fn, update_node_fn, mp_rounds,
                         nodes_init_fn, aggregate_edges)
    
    def _update_edges(self, nodes, edges, senders, receivers):
        sent_attributes = tree.tree_map(lambda n: n[:, senders], nodes)
        received_attributes = tree.tree_map(lambda n: n[:, receivers], nodes)
        
        feat_in = jnp.concatenate([
            edges,
            sent_attributes,
            received_attributes,
        ], axis=0)
        edges = self.update_edge_fn(feat_in)        
        return edges