import os
from functools import partial

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.ops.triton.layernorm import RMSNorm

from pearl.neural_networks.mamba.mamba_minimal import mamba as mamba_minimal
from pearl.neural_networks.mamba.mamba_parallel import mamba as mamba_parallel

from pearl.neural_networks.mamba.common.mamba_config import MambaConfig


class MambaWrapper(nn.Module):

    def __init__(self, observation_dim, action_dim, pscan, state_dim, num_mamba_layers, num_layers_per_block,
                 hidden_dim, device="cuda:0", project_first=True):
        super(MambaWrapper, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.parallel_scan = pscan
        self.state_dim = state_dim
        self.num_mamba_layers = num_mamba_layers
        self.num_layers_per_block = num_layers_per_block
        self.hidden_dim = hidden_dim
        self.device = device
        self.project_first = project_first

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

        self.layers_in = nn.ModuleList()
        self.layers_mid = nn.ModuleList()
        self.layers_out = nn.ModuleList()

        if self.config.input_dim != self.config.d_model:
            if not self.project_first:
                self.layers_in.extend(
                    self._create_mamba_layer(input_dim),
                )
                self.layers_in.append(
                    nn.Linear(self.config.input_dim * 2, self.config.d_model * 2)
                )
            else:
                self.layers_in = nn.Linear(self.config.input_dim, self.config.d_model)
        for i in range(self.num_mamba_layers):
            current_layer_idx = i * self.config.n_layers
            self.layers_mid.extend(self._create_mamba_layer(self.config.d_model, current_layer_idx))

        if self.config.d_model != self.config.state_dim:
            self.layers_mid.append(nn.Linear(self.config.d_model * 2, self.config.state_dim * 2))
            self.layers_out.extend(
                self._create_mamba_layer(self.config.state_dim)
            )

    def _create_mamba_layer(self, model_dim: int, layer_idx: int = None):
        mamba_layer = nn.ModuleList()

        if self.parallel_scan:
            # check if os is linux
            if os.name == 'posix':
                for j in range(self.config.n_layers):
                    ssm_layer_config = {
                        'conv_bias': self.config.conv_bias,
                        'bias': self.config.bias,
                    }
                    mamba_layer.append(
                        self._create_mamba_block(
                            model_dim,
                            ssm_cfg=ssm_layer_config,
                            layer_idx=layer_idx+j,
                            device=self.device,
                            dtype=torch.float32,
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

    def _create_mamba_block(
            self,
            d_model,
            ssm_cfg=None,
            norm_epsilon=1e-5,
            rms_norm=False,
            residual_in_fp32=False,
            fused_add_norm=False,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        if ssm_cfg is None:
            ssm_cfg = {}
        factory_kwargs = {"device": device, "dtype": dtype}
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )
        block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
        block.layer_idx = layer_idx
        return block

    def forward(self, x, resid_input, resid_mid, resid_out, inference_params=None):
        # if device is cuda, move to cuda
        if self.device == "cuda:0":
            x = x.cuda()

        # handle missing batch dimension
        if x.ndim == 2:
            x = x.unsqueeze(0)

        hid, resid_input = x, resid_input
        if self.config.input_dim != self.config.d_model:
            if not self.project_first:
                hid_in, resid_in = self._execute_layers(self.layers_in, hid, resid_input, inference_params=inference_params)
            else:
                # execute the linear layer
                hid_in = self.layers_in(hid)
                resid_in = resid_input
        else:
            hid_in = hid
            resid_in = resid_input
        hid_mid, resid_mid = self._execute_layers(self.layers_mid, hid_in, resid_in, inference_params=inference_params)
        hid_out, resid_out = self._execute_layers(self.layers_out, hid_mid, resid_mid, inference_params=inference_params)
        return hid_out, resid_in, resid_mid, resid_out

    def _execute_layers(self, layer_mod_list: nn.ModuleList, hid: torch.Tensor, resid: torch.Tensor,
                        inference_params=None):
        for i in range(len(layer_mod_list)):
            layer = layer_mod_list[i]
            if isinstance(layer, Block):
                hid, resid = layer(hid, resid, inference_params=inference_params)
            elif isinstance(layer, nn.Linear):
                if i == len(layer_mod_list) - 1:
                    # concat the input and residual
                    hid = torch.cat((hid, resid), dim=-1)
                hid = layer(hid)
                if i == len(layer_mod_list) - 1:
                    # split the output into hid and resid
                    # note that hid and resid have same shape
                    # so we split the last dim in half
                    hid, resid = torch.split(hid, hid.shape[-1] // 2, dim=-1)
            else:
                hid = layer(hid)
        return hid, resid
