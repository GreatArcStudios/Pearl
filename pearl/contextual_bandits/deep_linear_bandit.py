#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import Any, Dict, List

import torch

from pearl.api.action import Action

from pearl.api.action_space import ActionSpace
from pearl.contextual_bandits.deep_bandit import DeepBandit
from pearl.contextual_bandits.linear_bandit import LinearBandit
from pearl.contextual_bandits.linear_regression import AvgWeightLinearRegression
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.replay_buffer.transition import TransitionBatch


class DeepLinearBandit(DeepBandit):
    """
    Policy Learner for Contextual Bandit with:
    features --> neural networks --> linear regression --> predicted rewards
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int],  # last one is the input dim for linear regression
        exploration_module: ExplorationModule,
        training_rounds: int = 100,
        batch_size: int = 128,
        learning_rate: float = 0.001,
    ) -> None:
        assert (
            len(hidden_dims) >= 1
        ), "hidden_dims should have at least one value to specify feature dim for linear regression"
        DeepBandit.__init__(
            self,
            feature_dim=feature_dim,
            hidden_dims=hidden_dims[:-1],
            output_dim=hidden_dims[-1],
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
        )
        # TODO specify linear regression type when needed
        self._linear_regression = AvgWeightLinearRegression(feature_dim=hidden_dims[-1])
        self._linear_regression_dim = hidden_dims[-1]

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        input_features = torch.cat([batch.state, batch.action], dim=1)

        # forward pass
        mlp_output = self._deep_represent_layers(input_features)
        current_values = self._linear_regression(mlp_output)
        expected_values = batch.reward

        criterion = torch.nn.MSELoss()
        loss = criterion(current_values.view(expected_values.shape), expected_values)

        # Optimize the deep layer
        # TODO how should we handle weight in NN training
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # Optimize linear regression
        self._linear_regression.train(
            mlp_output.detach(), expected_values, batch.weight
        )
        return {"mlp_loss": loss.item()}

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        raise NotImplementedError("Implement when there is a usecase")

    def get_scores(
        self,
        subjective_state: SubjectiveState,
    ) -> torch.Tensor:
        processed_feature = self._deep_represent_layers(subjective_state)
        return LinearBandit.get_linucb_scores(
            subjective_state=processed_feature,
            feature_dim=self._linear_regression_dim,
            exploration_module=self._exploration_module,
            linear_regression=self._linear_regression,
        )
