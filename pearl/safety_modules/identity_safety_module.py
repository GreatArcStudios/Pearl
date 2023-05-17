from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.replay_buffer.replay_buffer import ReplayBuffer
from pearl.replay_buffer.transition import TransitionBatch
from pearl.safety_modules.safety_module import SafetyModule


class IdentitySafetyModule(SafetyModule):
    """
    A safety module that does not restrict action spaces.
    """

    def __init__(self, **options) -> None:
        self._action_space = None

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    def filter_action(self, subjective_state: SubjectiveState) -> ActionSpace:
        return self._action_space

    def learn(self, replay_buffer: ReplayBuffer) -> None:
        pass

    def learn_batch(self, batch: TransitionBatch) -> None:
        pass
