from typing import Callable

from jax import random
import jax.numpy as jnp
import equinox as eqx

from utils import graph_to_low_tri_mat
from data import bi_direc_edge_avg

class MLP(eqx.Module):
    layers: list
    act: Callable
        
    def __init__(self, features, N_layers, key, act):
        super(MLP, self).__init__()
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
        x = self.layers[:-1](x)
        return x

class PrecNet(eqx.Module):
    NodeEncoder: Callable
    EdgeEncoder: Callable
    MessagePass: Callable
    EdgeDecoder: Callable
    mp_rounds: int = eqx.field(static=True)
        
    def __init__(self, NodeEncoder, EdgeEncoder, MessagePass, EdgeDecoder, mp_rounds):
        super(PrecNet, self).__init__()
        @concatenated_args
        self.NodeEncoder = NodeEncoder
        @concatenated_args
        self.EdgeEncoder = EdgeEncoder
        @concatenated_args
        self.MessagePass = MessagePass
        @concatenated_args
        self.EdgeDecoder = EdgeDecoder
        self.mp_rounds = mp_rounds
        return
    
    def __call__(self, graph, bi_edges_indx):
        graph.nodes = self.NodeEncoder(graph.nodes)
        graph.edges = self.NodeEncoder(graph.edges)
        for _ in range(self.mp_rounds):
            graph = self.MessagePass(graph)
        graph = bi_direc_edge_avg(graph, bi_edges_indx)
        low_tri = graph_to_low_tri_mat(graph)
        return low_tri