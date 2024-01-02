import jax
import jax.numpy as np
from einops import rearrange, repeat
from flax import linen as nn
from flax.linen.initializers import normal as flax_normal
from jax.nn.initializers import lecun_normal, normal
from jax.scipy.linalg import block_diag
import math
from functools import partial

from .model import ResidualBlock, MambaBlock, ModelArgs
# from model import ResidualBlock, MambaBlock, ModelArgs

class MambaOperator(nn.Module):
    d_model: int
    n_layer: int
    l_max: int
    ssm_size: int = 64
    ssm_blocks: int = 1
    order: int = 2
    num_heads: int = 1
    inner_factor: int = 1
    num_blocks: int = 1
    fused_bias_fc: bool = False
    outer_mixing: bool = False
    drop_rate: float = 0.0
    filter_dropout: float = 0.0
    filter_cls: str = 'None'
    post_order_ffn: bool = False
    jit_filter: bool = False
    # short_filter_order: int = 3
    activation_type: str = "id"
    return_state: bool = False
    filter_args: dict = None

    hidden_state_method: str = None
    vocab_size: int = None
    d_state: int = None
    expand: int = None
    d_inner: int = None

    def setup(self):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            n_layer (int): # of model layers, (used for special scaled init)
            l_max: (int): Maximum input sequence length. Defaults to None
            ssm_size: (int): Size of the ssm
            ssm_blocks: (int): Number of initial blocks to use when initialzing SSM state matrix
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            num_heads: (int): Number of heads. Defaults to 1
            inner_factor: (int): Width multiplier. Defaults to 1
            num_blocks: (int): Number of blocks in sequence length. Defaults to 1
            fused_bias_fc: (bool): Whether to use fused bias FC. Defaults to False
            drop_rate: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
            post_order_ffn: (bool): Apply a dense layer between steps of the recurrence. Defaults to False
            jit_filter: (bool): Whether JIT the implicit filter function. Defaults to False
            short_filter_order: (int): Length of the explicit input convolutional filter. Defaults to 3, Removed
            activation_type: (str): type of act between kernel output and FF (default identity)
            return_state: (bool): whether to return a state
        """

        assert self.vocab_size is not None

        # print("n_layer is", self.n_layer, "But I'll just use n_layer=1")
        # self.args = ModelArgs(d_model=self.d_model,n_layer=self.n_layer, vocab_size=self.vocab_size)
        self.args = ModelArgs(d_model=self.d_model,n_layer=1, d_state=self.d_state, vocab_size=self.vocab_size)
        self.mamba = MambaBlock(self.args, hidden_state_method=self.hidden_state_method)

    @nn.compact
    def __call__(self, u, hiddens, training, layer_index=None):
        new_hiddens = []

        y, new_hiddens_stacked = self.mamba(u, hiddens)

        if self.return_state:
            return y, new_hiddens_stacked
        else:
            return y, None

    @property
    def d_output(self):
        return self.d_model


if __name__ == '__main__':
    d_model = 64
    hidden_state_method = "zero"
    vocab_size = 1024

    model = MambaOperator(d_model=d_model, n_layer=2, l_max=128, hidden_state_method=hidden_state_method, vocab_size=vocab_size)

    rng = jax.random.PRNGKey(0)
    input_shape = (1, 16, d_model)  # example shape
    inputs = jax.random.normal(rng, input_shape)


    # Initialize parameters
    params = model.init(rng, inputs, hiddens=None, training=True)

    # Apply the model
    outputs, hiddens = model.apply(params, inputs, hiddens=None, training=True)
    print(outputs.shape)
    assert outputs.shape == input_shape

#### 
    
    hidden_state_method = "previous"
    vocab_size = 1024
    d_state = 16
    expand = 2

    model = MambaOperator(d_model=d_model, n_layer=2, l_max=128, hidden_state_method=hidden_state_method, vocab_size=vocab_size, d_state=d_state)

    rng = jax.random.PRNGKey(0)
    input_shape = (1, 16, d_model)  # example shape
    inputs = jax.random.normal(rng, input_shape)

    hiddens = jax.random.normal(rng, (1, 1, d_model * expand, d_state))

    # Initialize parameters
    params = model.init(rng, inputs, hiddens=hiddens, training=True)

    # Apply the model
    outputs, hiddens = model.apply(params, inputs, hiddens=hiddens, training=True)
    print(outputs.shape)
    assert outputs.shape == input_shape

