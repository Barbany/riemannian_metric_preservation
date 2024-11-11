import torch
from torch import nn

from . import get_activation


class MLP(nn.Module):
    def __init__(
        self,
        activation,
        pos_enc,
        num_layers,
        hidden_size_first,
        hidden_size_mid,
        hidden_size_last,
        batch_norm,
        skips,
        **kwargs,
    ):
        super().__init__()
        self.pos_enc = pos_enc
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.skips = skips

        if self.pos_enc:
            from .embedder import Embedder

            self.embedder = Embedder(input_dims=2, **kwargs)
            in_fc = self.embedder.out_dim
        else:
            self.embedder = None
            in_fc = 2
        self.fc_first = nn.Linear(in_fc, hidden_size_first, bias=not self.batch_norm)

        if self.batch_norm:
            self.bn_first = nn.BatchNorm1d(hidden_size_first)

        fc_list = []
        bn_list = []
        for i in range(1, num_layers + 1):
            if i == 1:
                in_features = hidden_size_first
            else:
                in_features = hidden_size_mid
            if i in skips:
                in_features += in_fc

            if i == num_layers:
                out_features = hidden_size_last
            else:
                out_features = hidden_size_mid

            fc_list.append(
                nn.Linear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=not self.batch_norm,
                )
            )
            if self.batch_norm:
                bn_list.append(nn.BatchNorm1d(out_features))

        self.fc_list = nn.ModuleList(fc_list)
        if self.batch_norm:
            self.bn_list = nn.ModuleList(bn_list)

        self.fc_last = nn.Linear(
            hidden_size_last if i + 1 not in skips else hidden_size_last + in_fc,
            3,
        )

        self.activation_fn = get_activation(activation)

    def forward(self, x):
        if self.pos_enc:
            assert self.embedder is not None
            x = self.embedder(x)
        if self.batch_norm:
            h = self.activation_fn(self.bn_first(self.fc_first(x)))
        else:
            h = self.activation_fn(self.fc_first(x))

        for i in range(self.num_layers):
            if i + 1 in self.skips:
                h = torch.cat([x, h], -1)

            if self.batch_norm:
                h = self.activation_fn(self.bn_list[i](self.fc_list[i](h)))
            else:
                h = self.activation_fn(self.fc_list[i](h))

        if i + 2 in self.skips:
            h = torch.cat([x, h], -1)
        return self.fc_last(h)
