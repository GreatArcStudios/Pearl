import os

import torch.nn as nn

from mamba_ssm import Mamba
from pearl.neural_networks.mamba.mamba_minimal import mamba as mamba_minimal
from pearl.neural_networks.mamba.mamba_parallel import mamba as mamba_parallel

from pearl.neural_networks.mamba.common.mamba_config import MambaConfig


class MambaWrapper(nn.Module):

    def __init__(self, observation_dim, action_dim, pscan, state_dim, num_mamba_layers, num_layers_per_block,
                 hidden_dim, device="cuda:0"):
        super(MambaWrapper, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.parallel_scan = pscan
        self.state_dim = state_dim
        self.num_mamba_layers = num_mamba_layers
        self.num_layers_per_block = num_layers_per_block
        self.hidden_dim = hidden_dim
        self.device = device

        input_dim = self.observation_dim + self.action_dim
        if not self.hidden_dim:
            d_model = input_dim
        else:
            d_model = self.hidden_dim

        if not self.state_dim:
            state_dim = d_model

        self.config = MambaConfig(
            d_model=d_model,
            n_layers=self.num_layers_per_block,
            num_mamba_layers=self.num_mamba_layers,
            input_dim=input_dim,
            pscan=self.parallel_scan,
            state_dim=state_dim
        )

        # print out the mamba model args
        print(self.config)

        self.layers = nn.ModuleList()

        if self.config.input_dim != self.config.d_model:
            self.layers.extend(
                self._create_mamba_layer(input_dim),
                )
            self.layers.append(
                nn.Linear(self.config.input_dim, self.config.d_model)
            )
        for i in range(self.num_mamba_layers):
            self.layers.extend(self._create_mamba_layer(self.config.d_model))
            if i == self.num_mamba_layers - 1:
                self.layers.append(nn.Linear(self.config.d_model, self.config.state_dim))

        if self.config.d_model != self.config.state_dim:
            self.layers.extend(
                self._create_mamba_layer(self.config.state_dim)
            )

    def _create_mamba_layer(self, model_dim: int):
        mamba_layer = nn.ModuleList()

        if self.parallel_scan:
            # check if os is linux
            if os.name == 'posix':
                for i in range(self.config.n_layers):
                    mamba_layer.append(
                        Mamba(
                            d_model=model_dim,
                            conv_bias=self.config.conv_bias,
                            bias=self.config.bias,
                            device=self.device,
                        )
                    )
            else:
                mamba_model = mamba_parallel.Mamba(
                    self.config
                )
                mamba_layer.append(mamba_model)
        else:
            mamba_model = mamba_minimal.Mamba(
                self.config
            )
            mamba_layer.append(mamba_model)

        return mamba_layer

    def forward(self, x):
        # if device is cuda, move to cuda
        if self.device == "cuda:0":
            x = x.cuda()

        # handle missing batch dimension
        if x.ndim == 2:
            x = x.unsqueeze(0)

        hid = x
        for layer in self.layers:
            hid = layer(hid)

        return hid
