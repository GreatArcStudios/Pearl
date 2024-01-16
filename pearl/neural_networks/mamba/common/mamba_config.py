import math
from dataclasses import dataclass
from typing import Union


@dataclass
class MambaConfig:
    input_dim: int
    n_layers: int
    num_mamba_layers: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 1
    conv_bias: bool = True
    bias: bool = False
    tie_embeddings: bool = False
    use_minimal: bool = False
    d_model: int = None
    state_dim: int = None

    # parallel scan specific
    pscan: bool = True  #  use parallel scan mode or sequential mode when training
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  #  "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    def __post_init__(self):
        if self.d_model is None:
            self.d_model = self.input_dim
        if self.state_dim is None:
            self.state_dim = self.d_model

        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.use_minimal and self.input_dim % self.pad_vocab_size_multiple != 0:
            self.input_dim += (self.pad_vocab_size_multiple
                               - self.input_dim % self.pad_vocab_size_multiple)
