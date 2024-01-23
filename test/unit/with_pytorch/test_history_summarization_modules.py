# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import unittest

import torch
from pearl.history_summarization_modules.lstm_history_summarization_module import (
    LSTMHistorySummarizationModule,
)
from pearl.history_summarization_modules.mamba_history_summmarization_module import MambaHistorySummarizationModule
from pearl.history_summarization_modules.stacking_history_summarization_module import (
    StackingHistorySummarizationModule,
)


class TestHistorySummarizationModules(unittest.TestCase):
    def setUp(self) -> None:
        self.observation_dim: int = 10
        self.action_dim: int = 1
        self.history_length: int = 5

    def test_stacking_history_summarizer(self) -> None:
        """
        Easy test for stacking history summarization module.
        """
        summarization_module = StackingHistorySummarizationModule(
            self.observation_dim, self.action_dim, self.history_length
        )
        for _ in range(10):
            observation = torch.rand((1, self.observation_dim))
            action = torch.rand((1, self.action_dim))
            subjective_state = summarization_module.summarize_history(
                observation, action
            )
            self.assertEqual(
                subjective_state.shape[0],
                self.history_length * (self.action_dim + self.observation_dim),
            )

    def test_lstm_history_summarizer(self) -> None:
        """
        Easy test for LSTM history summarization module.
        """
        summarization_module = LSTMHistorySummarizationModule(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            history_length=self.history_length,
            hidden_dim=self.observation_dim + self.action_dim,
        )
        for _ in range(10):
            observation = torch.rand((1, self.observation_dim))
            action = torch.rand((1, self.action_dim))
            subjective_state = summarization_module.summarize_history(
                observation, action
            )
            self.assertEqual(
                subjective_state.shape[0], self.observation_dim + self.action_dim
            )

    def test_minimal_mamba_history_summarizer(self) -> None:
        """
        Easy test for minimal mamba (no parallel scan) history summarization module.
        """
        summarization_module = MambaHistorySummarizationModule(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            history_length=self.history_length,
            hidden_dim=self.observation_dim + self.action_dim,
            parallel_scan=False
        )
        for _ in range(10):
            observation = torch.rand((1, self.observation_dim))
            action = torch.rand((1, self.action_dim))
            subjective_state = summarization_module.summarize_history(
                observation, action
            )
            self.assertEqual(
                subjective_state.shape[0], self.observation_dim + self.action_dim
            )

    def test_parallel_mamba_history_summarizer(self) -> None:
        """
        Easy test for parallel mamba history summarization module.
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        state_dim = self.observation_dim + self.action_dim
        summarization_module = MambaHistorySummarizationModule(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            history_length=self.history_length,
            hidden_dim=state_dim,
            state_dim=state_dim,
            parallel_scan=True,
            device=device
        )
        for _ in range(10):
            observation = torch.rand((1, self.observation_dim))
            action = torch.rand((1, self.action_dim))
            subjective_state = summarization_module.summarize_history(
                observation, action
            )
            self.assertEqual(
                subjective_state.shape[0], state_dim
            )