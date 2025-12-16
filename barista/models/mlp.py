from typing import List

import torch.nn as nn

from barista.models.utils import get_activation_function


class MLP(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_out: int,
        layer_list: List = None,
        dropout: float = 0.1,
        bias: bool = True,
        use_first_dropout: bool = True,
        use_final_dropout: bool = False,
        use_final_activation: bool = False,
        activation: str = "linear",
        use_identity_stub: bool = True,
        **kwargs
    ):
        super(MLP, self).__init__()

        self.d_input = d_input
        self.d_out = d_out
        self.layer_list = layer_list
        self.dropout = dropout
        self.use_first_dropout = use_first_dropout
        self.use_final_dropout = use_final_dropout
        self.use_final_activation = use_final_activation
        self.activation_fn = get_activation_function(activation)

        current_dim = self.d_input
        self.layers = nn.ModuleList()
        if self.layer_list is not None:
            for _, dim in enumerate(self.layer_list):
                self.layers.append(nn.Linear(current_dim, dim, bias=bias))
                current_dim = dim
        else:
            if use_identity_stub:
                self.layers.append(nn.Identity())

        self.final_layer = nn.Linear(current_dim, self.d_out, bias=bias)

    def forward(self, x, *args, **kwargs):
        if self.use_first_dropout:
            x = nn.Dropout(self.dropout)(x)
        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)
            x = nn.Dropout(self.dropout)(x)
        x = self.final_layer(x)
        if self.use_final_activation:
            x = self.activation_fn(x)
        if self.use_final_dropout:
            x = nn.Dropout(self.dropout)(x)
        return x

    