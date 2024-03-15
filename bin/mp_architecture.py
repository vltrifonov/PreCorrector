# Original code of `GraphNetwork_NodeEdge` from the 'https://github.com/google-deepmind/jraph/blob/master/jraph/_src/models.py'.
# The only difference is that according to paper 'https://arxiv.org/abs/2305.16432',
# first we update node values, then edge values (original implementation updates edges first).

import functools
from typing import Any, Optional, Union, Iterable, Mapping, Optional, Callable

import jax
import jax.numpy as jnp
import jax.tree_util as tree

from jraph._src import graph as gn_graph
from jraph._src import utils

# As of 04/2020 pytype doesn't support recursive types.
# pytype: disable=not-supported-yet
ArrayTree = Union[jnp.ndarray, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]

# All features will be an ArrayTree.
NodeFeatures = EdgeFeatures = SenderFeatures = ReceiverFeatures = Globals = ArrayTree

# Signature:
# (edges of each node to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToNodesFn = Callable[
    [EdgeFeatures, jnp.ndarray, int], NodeFeatures]

# Signature:
# (nodes of each graph to be aggregated, segment ids, number of segments) ->
# aggregated nodes
AggregateNodesToGlobalsFn = Callable[[NodeFeatures, jnp.ndarray, int],
                                     Globals]

# Signature:
# (edges of each graph to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToGlobalsFn = Callable[[EdgeFeatures, jnp.ndarray, int],
                                     Globals]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# attention weights
AttentionLogitFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], ArrayTree]

# Signature:
# (edge features, weights) -> edge features for node update
AttentionReduceFn = Callable[[EdgeFeatures, ArrayTree], EdgeFeatures]

# Signature:
# (edges to be normalized, segment ids, number of segments) ->
# normalized edges
AttentionNormalizeFn = Callable[[EdgeFeatures, jnp.ndarray, int], EdgeFeatures]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# updated edge features
GNUpdateEdgeFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], EdgeFeatures]

# Signature:
# (node features, outgoing edge features, incoming edge features,
#  globals) -> updated node features
GNUpdateNodeFn = Callable[
    [NodeFeatures, SenderFeatures, ReceiverFeatures, Globals], NodeFeatures]

GNUpdateGlobalFn = Callable[[NodeFeatures, EdgeFeatures, Globals], Globals]
def GraphNetwork_NodeEdge(
    update_edge_fn: Optional[GNUpdateEdgeFn],
    update_node_fn: Optional[GNUpdateNodeFn],
    update_global_fn: Optional[GNUpdateGlobalFn] = None,
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils.segment_sum,
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils.segment_sum,
    attention_logit_fn: Optional[AttentionLogitFn] = None,
    attention_normalize_fn: Optional[AttentionNormalizeFn] = utils.segment_softmax,
    attention_reduce_fn: Optional[AttentionReduceFn] = None):
    """Returns a method that applies a configured GraphNetwork.

    This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

    There is one difference. For the nodes update the class aggregates over the
    sender edges and receiver edges separately. This is a bit more general
    than the algorithm described in the paper. The original behaviour can be
    recovered by using only the receiver edge aggregations for the update.

    In addition this implementation supports softmax attention over incoming
    edge features.

    Example usage::

    gn = GraphNetwork(update_edge_function,
    update_node_function, **kwargs)
    # Conduct multiple rounds of message passing with the same parameters:
    for _ in range(num_message_passing_steps):
      graph = gn(graph)

    Args:
    update_edge_fn: function used to update the edges or None to deactivate edge
      updates.
    update_node_fn: function used to update the nodes or None to deactivate node
      updates.
    update_global_fn: function used to update the globals or None to deactivate
      globals updates.
    aggregate_edges_for_nodes_fn: function used to aggregate messages to each
      node.
    aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
      globals.
    aggregate_edges_for_globals_fn: function used to aggregate the edges for the
      globals.
    attention_logit_fn: function used to calculate the attention weights or
      None to deactivate attention mechanism.
    attention_normalize_fn: function used to normalize raw attention logits or
      None if attention mechanism is not active.
    attention_reduce_fn: function used to apply weights to the edge features or
      None if attention mechanism is not active.

    Returns:
    A method that applies the configured GraphNetwork.
    """
    not_both_supplied = lambda x, y: (x != y) and ((x is None) or (y is None))
    if not_both_supplied(attention_reduce_fn, attention_logit_fn):
        raise ValueError(('attention_logit_fn and attention_reduce_fn must both be'
                          ' supplied.'))

    def _ApplyGraphNet(graph):
        """Applies a configured GraphNetwork to a graph.

        This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

        There is one difference. For the nodes update the class aggregates over the
        sender edges and receiver edges separately. This is a bit more general
        the algorithm described in the paper. The original behaviour can be
        recovered by using only the receiver edge aggregations for the update.

        In addition this implementation supports softmax attention over incoming
        edge features.

        Many popular Graph Neural Networks can be implemented as special cases of
        GraphNets, for more information please see the paper.

        Args:
          graph: a `GraphsTuple` containing the graph.

        Returns:
          Updated `GraphsTuple`.
        """
        # pylint: disable=g-long-lambda
        nodes, edges, receivers, senders, globals_, n_node, n_edge = graph
        # Equivalent to jnp.sum(n_node), but jittable
        sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
        sum_n_edge = senders.shape[0]
        if not tree.tree_all(tree.tree_map(lambda n: n.shape[0] == sum_n_node, nodes)):
            raise ValueError('All node arrays in nest must contain the same number of nodes.')

        sent_attributes = tree.tree_map(lambda n: n[senders], nodes)
        received_attributes = tree.tree_map(lambda n: n[receivers], nodes)
        # Here we scatter the global features to the corresponding edges,
        # giving us tensors of shape [num_edges, global_feat].
        global_edge_attributes = tree.tree_map(lambda g: jnp.repeat(
            g, n_edge, axis=0, total_repeat_length=sum_n_edge), globals_)

#         if update_edge_fn:
#             edges = update_edge_fn(edges, sent_attributes, received_attributes,
#                                  global_edge_attributes)

        if attention_logit_fn:
            logits = attention_logit_fn(edges, sent_attributes, received_attributes,
                                      global_edge_attributes)
            tree_calculate_weights = functools.partial(
              attention_normalize_fn,
              segment_ids=receivers,
              num_segments=sum_n_node)
            weights = tree.tree_map(tree_calculate_weights, logits)
            edges = attention_reduce_fn(edges, weights)

        if update_node_fn:
            sent_attributes = tree.tree_map(
              lambda e: aggregate_edges_for_nodes_fn(e, senders, sum_n_node), edges)
            received_attributes = tree.tree_map(
              lambda e: aggregate_edges_for_nodes_fn(e, receivers, sum_n_node),
              edges)
            # Here we scatter the global features to the corresponding nodes,
            # giving us tensors of shape [num_nodes, global_feat].
            global_attributes = tree.tree_map(lambda g: jnp.repeat(
              g, n_node, axis=0, total_repeat_length=sum_n_node), globals_)
            nodes = update_node_fn(nodes, sent_attributes,
                                 received_attributes, global_attributes)
        # !!!
        if update_edge_fn:
            edges = update_edge_fn(edges, sent_attributes, received_attributes,
                                 global_edge_attributes)
            
        if update_global_fn:
            n_graph = n_node.shape[0]
            graph_idx = jnp.arange(n_graph)
            # To aggregate nodes and edges from each graph to global features,
            # we first construct tensors that map the node to the corresponding graph.
            # For example, if you have `n_node=[1,2]`, we construct the tensor
            # [0, 1, 1]. We then do the same for edges.
            node_gr_idx = jnp.repeat(
              graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
            edge_gr_idx = jnp.repeat(
              graph_idx, n_edge, axis=0, total_repeat_length=sum_n_edge)
            # We use the aggregation function to pool the nodes/edges per graph.
            node_attributes = tree.tree_map(
              lambda n: aggregate_nodes_for_globals_fn(n, node_gr_idx, n_graph),
              nodes)
            edge_attribtutes = tree.tree_map(
              lambda e: aggregate_edges_for_globals_fn(e, edge_gr_idx, n_graph),
              edges)
            # These pooled nodes are the inputs to the global update fn.
            globals_ = update_global_fn(node_attributes, edge_attribtutes, globals_)
        # pylint: enable=g-long-lambda
        return gn_graph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            receivers=receivers,
            senders=senders,
            globals=globals_,
            n_node=n_node,
            n_edge=n_edge)

    return _ApplyGraphNet