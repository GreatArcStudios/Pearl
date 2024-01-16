# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional

import torch
from pearl.api.action import Action
from pearl.api.observation import Observation
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
)
from pearl.neural_networks.mamba.common.mamba_config import MambaConfig
from pearl.neural_networks.mamba.common.mamba_wrapper import MambaWrapper
from pearl.neural_networks.mamba.mamba_minimal import mamba as mamba_minimal
from pearl.neural_networks.mamba.mamba_parallel import mamba as mamba_parallel


class MambaHistorySummarizationModule(HistorySummarizationModule):
    """
    A history summarization module that uses a recurrent neural network
    to summarize past history observations into a hidden representation
    and incrementally generate a new subjective state.
    """

    def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            history_length: int = 8,
            hidden_dim: int = None,
            state_dim: int = None,
            num_layers: int = 2,
            parallel_scan: bool = True,
            device: str = "cuda:0",
    ) -> None:
        super(MambaHistorySummarizationModule, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.history_length = history_length
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.parallel_scan = parallel_scan
        self.register_buffer("default_action", torch.zeros((1, action_dim)))
        self.register_buffer(
            "history",
            torch.zeros((self.history_length, self.action_dim + self.observation_dim)),
        )

        self.mamba = MambaWrapper(
            observation_dim=observation_dim,
            action_dim=action_dim,
            pscan=parallel_scan,
            state_dim=state_dim,
            num_mamba_layers=3,
            num_layers_per_block=num_layers,
            hidden_dim=hidden_dim,
            device=device
        )

        if device == "cuda:0":
            self.mamba.cuda()

        # input_dim = observation_dim + action_dim
        # if not hidden_dim:
        #     d_model = input_dim
        # else:
        #     d_model = hidden_dim
        #
        # if not state_dim:
        #     state_dim = d_model
        #
        # self.mamba_model_args = MambaConfig(
        #     d_model=d_model,
        #     n_layers=num_layers,
        #     input_dim=input_dim,
        #     pscan=parallel_scan,
        #     state_dim=state_dim
        # )
        #
        # # print out the mamba model args
        # print(self.mamba_model_args)
        #
        # if parallel_scan:
        #     mamba_model = mamba_parallel.Mamba(
        #         self.mamba_model_args
        #     )
        # else:
        #     mamba_model = mamba_minimal.Mamba(
        #         self.mamba_model_args
        #     )
        #
        # self.mamba = mamba_model

    def summarize_history(
            self, observation: Observation, action: Optional[Action]
    ) -> torch.Tensor:
        assert isinstance(observation, torch.Tensor)
        observation = (
            observation.clone().detach().float().view((1, self.observation_dim))
        )
        if action is None:
            action = self.default_action
        assert isinstance(action, torch.Tensor)
        action = action.clone().detach().float().view((1, self.action_dim))
        observation_action_pair = torch.cat((action, observation.view(1, -1)), dim=-1)

        assert observation.shape[-1] + action.shape[-1] == self.history.shape[-1]
        self.history = torch.cat(
            [
                self.history[1:, :],
                observation_action_pair.view(
                    (1, self.action_dim + self.observation_dim)
                ),
            ],
            dim=0,
        )
        out_batched = self.mamba(self.history)
        out_no_batch = out_batched.squeeze(0)
        out_final = out_no_batch[-1, :].view((1, -1))
        return out_final.squeeze(0)

    def get_history(self) -> torch.Tensor:
        return self.history

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        out = self.mamba(x)
        return out[:, -1, :].view((batch_size, -1))

    def reset(self) -> None:
        self.register_buffer(
            "history",
            torch.zeros((self.history_length, self.action_dim + self.observation_dim)),
        )
