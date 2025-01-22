from typing import Callable, Union
from collections.abc import Sequence
from jaxtyping import PRNGKeyArray

import jax
from jax import random
import jax.nn as jnn
import equinox as eqx

class FullyConnected(eqx.Module):
    layers: list
    act: Callable = eqx.field(static=True)
        
    def __init__(self, features, N_layers, key, act=jnn.relu, layer_=eqx.nn.Conv1d):
        super(FullyConnected, self).__init__()
        N_in, N_pr, N_out = features
        keys = random.split(key, N_layers)
        Ns = [N_in,] + [N_pr,] * (N_layers - 1) + [N_out,]
        self.layers = [layer_(in_channels=N_in, out_channels=N_out, kernel_size=1, key=key) for N_in, N_out, key in zip(Ns[:-1], Ns[1:], keys)]
        self.act = act
        return
    
    def __call__(self, x):
        for l in self.layers[:-1]:
            x = l(x)
            x = self.act(x)
        x = self.layers[-1](x)
        return x
    
class DummyLayer(eqx.Module):
    'Do nothing. Returns second channel of the input'
    def __init__(self, *args, **kwargs):
        super(DummyLayer, self).__init__()
        return
        
    def __call__(self, x, *args, **kwargs):
        return x[1:2, ...]
    
class ConstantConv1d(eqx.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int], Sequence[tuple[int, int]]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        const: float = 0,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         use_bias=use_bias,
                         key=key)
        self.weight = self.weight * const
        if self.use_bias:
            self.bias = self.bias * const
        return