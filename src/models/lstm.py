import jax
import jax.numpy as np
from einops import rearrange, repeat
from flax import linen as nn
from flax.linen.initializers import normal as flax_normal
from jax.nn.initializers import lecun_normal, normal
from jax.scipy.linalg import block_diag
import math
from functools import partial

from .hyena import Activation, mul_sum
from .SSM_init import init_CV, init_VinvB,\
    make_DPLR_HiPPO, init_log_steps, trunc_standard_normal


def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, hiddens, conj_sym, hidden_state_method=None, key=None):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            hiddens        (float32): hidden state                       (1, H)
            conj_sym (bool):         whether conjugate symmetry is enforced
            hidden_state_method (str): Which hidden state method used, zero or previous or random. Defaults to None
            key (jax.random.PRNGKey): random key for random hidden state initialization. Defaults to None

        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    Lambda_elements = Lambda_bar * np.ones((input_sequence.shape[0] + 1,
                                            Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

    if hidden_state_method == "previous":
        # print("Using previous hidden state")
        # print("In apply_ssm, hiddens shape", hiddens.shape)
        assert hiddens is not None
        assert len(hiddens.shape) == 2
        Bu_elements = np.vstack([hiddens, Bu_elements])
    elif hidden_state_method == "zero":
        # print("Using zero init hidden state")
        hiddens = jax.numpy.zeros((1, Bu_elements.shape[1]))
        Bu_elements = np.vstack([hiddens, Bu_elements])
    elif hidden_state_method == "trueRandom":
        assert key is not None
        key, subkey = jax.random.split(key)
        scale_factor = np.abs(Bu_elements[0,:]).max()

        print("Subkey is", subkey)
        hiddens = jax.random.normal(subkey, (1, Bu_elements.shape[1])) * scale_factor
        Bu_elements = np.vstack([hiddens, Bu_elements])
    else:
        raise NotImplementedError(
            "hidden_state_method {} not implemented".format(hidden_state_method)
        )

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    if conj_sym:
        return jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs[1:, :]), xs[-1:, :]
    else:
        return jax.vmap(lambda x: (C_tilde @ x).real)(xs[1:, :]), xs[-1:, :]


def apply_ssm_with_feedthrough(input_sequence, hiddens, Lambda_bar, B_bar, C_tilde, D, conj_sym, key, hidden_state_method):
    ys, hiddens = apply_ssm(Lambda_bar,
                   B_bar,
                   C_tilde,
                   input_sequence,
                   hiddens, 
                   conj_sym,
                   hidden_state_method=hidden_state_method,
                   key=key,
                   )

    # Add feedthrough matrix output Du;
    Du = jax.vmap(lambda u: D * u)(input_sequence)
    output_sequence = ys + Du
    return output_sequence, hiddens


class S5SSM(nn.Module):
    Lambda_re_init: np.DeviceArray
    Lambda_im_init: np.DeviceArray
    V: np.DeviceArray
    Vinv: np.DeviceArray

    H: int
    P: int
    C_init: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    activation: str = "gelu"

    hidden_state_method: str = None
    rng_collection: str = "hidden_state"

    """ The S5 SSM
        Args:
            Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
            Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
            V           (complex64): Eigenvectors used for init           (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
            H           (int32):     Number of features of input seq 
            P           (int32):     state size
            C_init      (string):    Specifies How C is initialized
                         Options: [trunc_standard_normal: sample from truncated standard normal 
                                                        and then multiply by V, i.e. C_tilde=CV.
                                   lecun_normal: sample from Lecun_normal and then multiply by V.
                                   complex_normal: directly sample a complex valued output matrix 
                                                    from standard normal, does not multiply by V]
            dt_min:      (float32): minimum value to draw timescale values from when 
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when 
                                    initializing log_step
            conj_sym    (bool):    Whether conjugate symmetry is enforced
            clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                   constrain real part of eigenvalues to be negative. 
                                   True recommended for autoregressive task/unbounded sequence lengths
                                   Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            activation   (str):    type of activation to apply to SSM outputs

            hidden_state_method (str): Which hidden state method used, zero or previous or random. Defaults to None
            rng_collection (str): Which rng collection to use for hidden state initialization. Defaults to "hidden_state"

    """

    def setup(self):
        """Initializes parameters once and performs discretization each time
           the SSM is applied to a sequence
        """

        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2*self.P
        else:
            local_P = self.P

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        B_shape = (local_P, self.H)
        self.B = self.param("B",
                            lambda rng, shape: init_VinvB(B_init,
                                                          rng,
                                                          shape,
                                                          self.Vinv),
                            B_shape)
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                   "C_init method {} not implemented".format(self.C_init))

        if self.C_init in ["complex_normal"]:
            self.C = self.param("C", C_init, (self.H, self.P, 2))

        else:
            self.C = self.param("C",
                                lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                C_shape)

        self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = self.param("D", normal(stddev=1.0), (self.H,))

        # Initialize learnable discretization timescale value
        self.log_step = self.param("log_step",
                                   init_log_steps,
                                   (self.P, self.dt_min, self.dt_max))
        step = np.exp(self.log_step[:, 0])

        # Define activations
        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.H)
            self.out2 = nn.Dense(self.H)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.H)

        # Discretize
        self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)

    def __call__(self, input_sequence, hiddens, training=True, layer_index=None):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (bsz, num_heads, H, num_blocks, L)
                                       where for now num_heads and num_blocks is 1
        Returns:
            output sequence (float32): (bsz, num_heads, H, num_blocks, L)
        """

        input_sequence = input_sequence[:, 0, :, 0]
        input_sequence = input_sequence.transpose(0, 2, 1)
        # input sequence is now bsz, L, H
        # hiddens are bsz, 1, H
        
        ys, new_hiddens = jax.vmap(partial(apply_ssm_with_feedthrough, hidden_state_method=self.hidden_state_method),
                      in_axes=(0, 0, None, None, None, None, None, None)
                      )(input_sequence,
                        hiddens,
                        self.Lambda_bar,
                        self.B_bar,
                        self.C_tilde,
                        self.D,
                        self.conj_sym,
                        self.make_rng(self.rng_collection),
                        )  # ys is bsz, L, H
        
        # if layer_index < 1:
        #     print("In S5SSM, old hidden shape", hiddens.shape)
        #     print("In S5SSM, new hidden shape", new_hiddens.shape)
        assert new_hiddens.shape == hiddens.shape

        if self.activation in ["full_glu"]:
            ys = nn.activation.gelu(ys, approximate=False)
            ys = self.out1(ys) * jax.nn.sigmoid(self.out2(ys))
        elif self.activation in ["half_glu1"]:
            ys = nn.activation.gelu(ys, approximate=False)
            ys = ys * jax.nn.sigmoid(self.out2(ys))
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = nn.activation.gelu(ys, approximate=False)
            ys = ys * jax.nn.sigmoid(self.out2(x1))
        elif self.activation in ["gelu"]:
            ys = nn.activation.gelu(ys, approximate=False)
        else:
            raise NotImplementedError(
                "Activation: {} not implemented".format(self.activation))
        # ys is bsz, L, H
        output_sequence = np.expand_dims(ys.transpose(0, 2, 1), (1, 3))
        # output sequence is bsz, 1, H, 1, L

        return output_sequence, new_hiddens


def init_S5SSM(d_model, ssm_size, blocks, ssm_args, hidden_state_method):
    """Convenience function that will be used to initialize the SSM.
       Same arguments as defined in S5SSM above."""

    block_size = int(ssm_size / blocks)
    Lambda, _, _, V, _ = make_DPLR_HiPPO(block_size)

    if ssm_args['conj_sym']:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
    V = block_diag(*([V] * blocks))
    Vinv = block_diag(*([Vc] * blocks))

    # print("Lambda.shape={}".format(Lambda.shape))
    # print("V.shape={}".format(V.shape))
    # print("Vinv.shape={}".format(Vinv.shape))

    return S5SSM(Lambda.real,
                 Lambda.imag,
                 V,
                 Vinv,
                 H=d_model,
                 P=ssm_size,
                 hidden_state_method=hidden_state_method,
                 **ssm_args)





class LSTM_Operator(nn.Module):
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
    short_filter_order: int = 3
    activation_type: str = "id"
    return_state: bool = True
    filter_args: dict = None

    hidden_state_method: str = None
    vocab_size : int = None

    def setup(self):
        
        self.layer = nn.LSTMCell(features=self.d_model)

    @nn.compact
    def __call__(self, x, carry, training=None, layer_index=None):
        if self.hidden_state_method == "previous":
            assert carry is not None
            # carry = np.squeeze(carry, axis=1)
            assert len(carry.shape) == 3, f"len(carry.shape)={len(carry.shape)} != 3, carry.shape={carry.shape}"
        elif self.hidden_state_method == "zero":
            carry = self.layer.initialize_carry(jax.random.key(1), x[:,0,:].shape) # make it zero
            carry = np.stack(carry, axis=-1) # B * D * 2
            if layer_index < 1:
                print("In LSTM_Operator, this should be zero carry", carry)
        else:
            raise NotImplementedError(f"hidden_state_method {self.hidden_state_method} not implemented")

        T = x.shape[-2]

        assert len(carry.shape) == 3, f"len(carry.shape)={len(carry.shape)} != 3"
        # print("carry[0].shape is", carry[0].shape)
        carry = (carry[:,:,0], carry[:,:,1])

        # TODO, maybe convert into scan a bit later
        out_list = []
        for i in range(T):
            new_carry, out = self.layer(carry, x[:,i,:])
            out_list.append(out) # list of B * D
        out_list = np.stack(out_list, axis=-2) # B * T * D
        assert len(out_list.shape) == 3, f"len(out_list.shape)={len(out_list.shape)} != 3"
        assert out_list.shape == x.shape, f"out_list.shape={out_list.shape} != x.shape={x.shape}"

        new_carry = np.stack(new_carry, axis=-1) # B * D * 2

        if self.return_state:
            return out_list, new_carry
        else:
            return out_list, None

    def initialize_carry(self, key=jax.random.key(1), shape=None):
        """carry = layer.initialize_carry(jax.random.key(1), x.shape)"""

        assert shape is not None, "shape must be specified"
        
        return self.layer.initialize_carry(key, shape)

    @property
    def d_output(self):
        return self.d_model
