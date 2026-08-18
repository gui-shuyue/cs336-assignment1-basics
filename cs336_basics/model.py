from __future__ import annotations

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    """A linear transformation without a bias term.

    The weight follows PyTorch's convention and has shape
    ``(out_features, in_features)``. Inputs may have any number of leading
    dimensions, but their final dimension must equal ``in_features``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # nn.Parameter registers the tensor as a trainable model parameter, so
        # it appears in parameters()/state_dict() and receives gradients.
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        sigma = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=sigma,
            a=-3 * sigma,
            b=3 * sigma,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Transform ``(..., in_features)`` into ``(..., out_features)``."""
        return x @ self.weight.T


class Embedding(nn.Module):
    """Map integer token IDs to trainable embedding vectors.

    ``weight`` is a lookup table with shape ``(num_embeddings, embedding_dim)``.
    For an input with shape ``(...)``, the output has shape
    ``(..., embedding_dim)``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Row i stores the vector for token ID i. nn.Parameter makes the
        # complete lookup table trainable and registers it in the module.
        self.weight = nn.Parameter(
            torch.empty(
                (num_embeddings, embedding_dim),
                device=device,
                dtype=dtype,
            )
        )

        # TODO: Initialize self.weight from a normal distribution with
        # mean 0 and standard deviation 1. Inspect torch.nn.init.normal_.
        nn.init.normal_(self.weight, mean=0.0, std=1.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Look up vectors for token IDs with shape ``(...)``."""
        # TODO: Use token_ids as row indices into self.weight. Do not use
        # nn.Embedding or construct one-hot vectors. The result should have
        # shape token_ids.shape + (self.embedding_dim,).
        return self.weight[token_ids]


class SiLU(nn.Module):
    """Apply the SiLU activation independently to every tensor element.

    SiLU has no trainable parameters and preserves the input shape. Its
    mathematical definition is ``SiLU(x) = x * sigmoid(x)``.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Return an element-wise activation with the same shape as ``x``."""
        # TODO: Implement x * sigmoid(x) using basic PyTorch tensor
        # operations. Inspect torch.sigmoid. Do not call nn.SiLU or F.silu.
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """A gated feed-forward network built from three bias-free Linear layers.

    The two input branches project from ``d_model`` to ``d_ff``. Their
    element-wise gated product is then projected back to ``d_model``.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Assigning Modules to attributes registers them as child modules.
        # Their parameters will be discovered recursively by parameters(),
        # state_dict(), optimizers, and device/dtype conversion methods.
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """Transform ``(..., d_model)`` into ``(..., d_model)``."""
        # TODO: Follow the SwiGLU computation in three stages:
        # 1. Apply w1 followed by SiLU to produce the gate (..., d_ff).
        # 2. Apply w3 to produce the value branch (..., d_ff).
        # 3. Multiply both branches element-wise, then apply w2.
        # Do not use torch.matmul for the gate/value combination.
        gate = self.silu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)


class RMSNorm(nn.Module):
    """Normalize each token by its root-mean-square feature magnitude.

    The final dimension is treated as the feature dimension. All leading
    dimensions are preserved, and a trainable weight scales each feature.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # RMSNorm starts as pure normalization. Training can then learn a
        # separate scale for each of the d_model feature dimensions.
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``(..., d_model)`` and preserve its shape and dtype."""
        input_dtype = x.dtype

        # Compute the normalization statistics in float32 for stability,
        # especially when model activations use float16 or bfloat16.
        x_float = x.to(torch.float32)

        # TODO: Complete RMSNorm in four steps:
        # 1. Compute mean(x_float ** 2) over only the last dimension. Use
        #    keepdim=True so the result broadcasts back over d_model.
        # 2. Add self.eps and compute its reciprocal square root. Inspect
        #    torch.rsqrt.
        # 3. Multiply x_float by that inverse RMS, then by self.weight.
        # 4. Convert the result back to input_dtype before returning it.
        mean_square = torch.mean(x_float ** 2, dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + self.eps)
        normalized = x_float * inv_rms * self.weight
        return normalized.to(input_dtype)


class RotaryPositionalEmbedding(nn.Module):
    """Apply rotary positional embeddings (RoPE) to queries or keys.

    The final dimension is split into adjacent pairs. Each pair is rotated by
    a position-dependent angle, while all leading dimensions are preserved.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"RoPE requires an even d_k, got {d_k}")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inverse_frequencies = self._build_inverse_frequencies(
            theta=theta,
            d_k=d_k,
            device=device,
        )
        angle_table = self._build_angle_table(
            max_seq_len=max_seq_len,
            inverse_frequencies=inverse_frequencies,
            device=device,
        )

        # These tensors are model state but are not trainable Parameters.
        # persistent=False keeps deterministic caches out of state_dict().
        self.register_buffer(
            "cos_cache",
            torch.cos(angle_table),
            persistent=False,
        )
        self.register_buffer(
            "sin_cache",
            torch.sin(angle_table),
            persistent=False,
        )

    @staticmethod
    def _build_inverse_frequencies(
        theta: float,
        d_k: int,
        device: torch.device | str | None,
    ) -> Tensor:
        """Return one inverse frequency for each adjacent dimension pair."""
        # TODO 1A: Create the even feature indices 0, 2, ..., d_k - 2 with
        # torch.arange. Use float32 and place the tensor on device.
        # Expected shape: (d_k / 2,).
        indices = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        # TODO 1B: Divide those indices by d_k to obtain the exponents
        # 0/d_k, 2/d_k, ..., (d_k - 2)/d_k.
        exponents = indices / d_k
        # TODO 1C: Compute 1 / theta**exponents. Each dimension pair then
        # rotates at a different rate. Return a float32 tensor with shape
        # (d_k / 2,).
        return 1.0 / (theta ** exponents)

    @staticmethod
    def _build_angle_table(
        max_seq_len: int,
        inverse_frequencies: Tensor,
        device: torch.device | str | None,
    ) -> Tensor:
        """Return all position/frequency angle combinations."""
        # TODO 2A: Create positions 0, 1, ..., max_seq_len - 1 as float32 on
        # device. Expected shape: (max_seq_len,).
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        # TODO 2B: Form the outer product between positions and
        # inverse_frequencies. You may use torch.outer or add singleton
        # dimensions and multiply. Expected shape:
        # (max_seq_len, d_k / 2).
        return torch.outer(positions, inverse_frequencies)

    def _lookup_trig_values(
        self,
        token_positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Select cached cosine and sine rows for the requested positions."""
        # TODO 3A: Use token_positions as integer row indices into cos_cache
        # and sin_cache. If token_positions has shape (..., sequence_length),
        # each result should have shape (..., sequence_length, d_k / 2).
        position = token_positions.to(device=self.cos_cache.device, dtype=torch.long)
        # TODO 3B: Return the selected cosine and sine tensors as a tuple.
        return self.cos_cache[position], self.sin_cache[position]

    @staticmethod
    def _align_for_broadcast(
        trig_values: Tensor,
        target_ndim: int,
    ) -> Tensor:
        """Insert missing head/batch axes before sequence and pair axes."""
        # TODO 4: While trig_values has fewer dimensions than the query/key,
        # insert singleton dimensions immediately before its final two axes.
        # Tensor.unsqueeze(-3) is useful here. For example:
        #   (seq, pairs)       -> (1, 1, seq, pairs) for a 4-D target
        #   (batch, seq, pairs)-> (batch, 1, seq, pairs) for a 4-D target
        # These size-1 axes let PyTorch broadcasting cover attention heads.
        while trig_values.ndim < target_ndim:
            trig_values = trig_values.unsqueeze(-3)
        return trig_values

    @staticmethod
    def _rotate_pairs(
        x_float: Tensor,
        cosine: Tensor,
        sine: Tensor,
    ) -> Tensor:
        """Rotate every adjacent pair in the final dimension."""
        # TODO 5A: Split x_float into even and odd coordinates with slicing:
        # (..., d_k) -> two tensors of shape (..., d_k / 2).
        even = x_float[..., 0::2]
        odd = x_float[..., 1::2]
        # TODO 5B: Apply the two-dimensional rotation equations:
        # rotated_even = even * cosine - odd * sine
        # rotated_odd  = even * sine + odd * cosine
        rotated_even = even * cosine - odd * sine
        rotated_odd = even * sine + odd * cosine
        # TODO 5C: Interleave the rotated coordinates again. torch.stack
        # with dim=-1 produces (..., d_k / 2, 2); flattening the final two
        # dimensions restores (..., d_k) in even, odd, even, odd order.
        stack = torch.stack((rotated_even, rotated_odd), dim=-1)
        return stack.flatten(-2)

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor,
    ) -> Tensor:
        """Rotate ``(..., sequence_length, d_k)`` using token positions."""
        if x.shape[-1] != self.d_k:
            raise ValueError(
                f"Expected final dimension {self.d_k}, got {x.shape[-1]}"
            )

        input_dtype = x.dtype
        x_float = x.to(torch.float32)

        # TODO 6A: Call _lookup_trig_values(token_positions).
        cosine, sine = self._lookup_trig_values(token_positions)
        # TODO 6B: Align both returned tensors to x_float.ndim with
        # _align_for_broadcast.
        cosine = self._align_for_broadcast(cosine, x_float.ndim)
        sine = self._align_for_broadcast(sine, x_float.ndim)
        # TODO 6C: Call _rotate_pairs with x_float and the aligned tensors.
        x_rotated = self._rotate_pairs(x_float, cosine, sine)
        # TODO 6D: Convert the rotated result back to input_dtype and return.
        return x_rotated.to(input_dtype)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute scaled dot-product attention over arbitrary leading axes.

    Shapes:
        query: ``(..., queries, d_k)``
        key: ``(..., keys, d_k)``
        value: ``(..., keys, d_v)``
        mask: ``(..., queries, keys)`` where True means "may attend"
        output: ``(..., queries, d_v)``
    """
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            "Query and key must have the same final dimension, got "
            f"{query.shape[-1]} and {key.shape[-1]}"
        )
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(
            "Key and value must contain the same number of tokens, got "
            f"{key.shape[-2]} and {value.shape[-2]}"
        )

    d_k = query.shape[-1]

    # TODO 1: Transpose only the final two dimensions of key, then perform
    # batched matrix multiplication with query. Do not use key.T because .T
    # reverses all dimensions for tensors with more than two dimensions.
    # Expected score shape: (..., queries, keys).
    scores = torch.matmul(query, key.transpose(-2, -1))

    # TODO 2: Divide every score by sqrt(d_k). This prevents the dot products
    # from growing with the feature dimension and saturating softmax.
    scores = scores / d_k**0.5
    # TODO 3: If mask is not None, replace every score whose mask value is
    # False with negative infinity. Inspect Tensor.masked_fill and remember
    # that ~mask negates a boolean mask. Apply this before softmax.
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    # TODO 4: Apply softmax over the keys dimension, which is the final
    # dimension of scores. Each query's weights should sum to one.
    attention_weights = torch.softmax(scores, dim=-1)
    # TODO 5: Multiply the attention weights by value. The keys dimension is
    # contracted, producing (..., queries, d_v), and return the result.
    return attention_weights @ value


class MultiheadSelfAttention(nn.Module):
    """
    1. 生成Q/K/V
    2. 拆分head
    3. 应用RoPE
    4. Attention
    5. 合并head
    6. 输出
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if (theta is None) != (max_seq_len is None):
            raise ValueError("theta and max_seq_len must be provided together")
        self.rope = (
            RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device,
            )
            if theta is not None and max_seq_len is not None
            else None
        )

    def _split_heads(self, x):
        """Split the final dimension into (num_heads, d_k) and transpose."""
        reshaped = x.reshape(*x.shape[:-1], self.num_heads, self.d_k)  # (..., num_heads, d_k)
        return reshaped.transpose(-3, -2)  # (..., num_heads, seq_len, d_k)

    def _merge_heads(self, x):
        """Transpose and merge the final two dimensions into d_model."""
        transposed = x.transpose(-3, -2)  # (..., seq_len, num_heads, d_k)
        return transposed.reshape(*transposed.shape[:-2], self.d_model)  # (..., seq_len, d_model)

    def _make_causal_mask(self, seq_len: int, device: torch.device):
        """Create a causal mask for self-attention."""
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
        return mask  # shape: (seq_len, seq_len)

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        """Compute multihead self-attention with RoPE.

        Args:
            x: Float[Tensor, " ... sequence_length d_model"]
            token_positions: Long[Tensor, " ... sequence_length"]

        Returns:
            Float[Tensor, " ... sequence_length d_model"]: Output of MHA.
        """
        # 1. Generate Q/K/V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. Split heads
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # 3. Apply RoPE when this attention module was configured to use it.
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # 4. Attention
        seq_len = x.shape[-2]
        causal_mask = self._make_causal_mask(seq_len, x.device)
        output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # 5. Merge heads
        output = self._merge_heads(output)

        # 6. Mix information from all heads in the model dimension.
        return self.output_proj(output)


class TransformerBlock(nn.Module):
    """A single transformer block with self-attention and feed-forward layers."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,  # SwiGLU
        theta: float | None = None,
        max_seq_len: int | None = None,
        eps: float = 1e-5, # RMSNorm
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """Apply a transformer block to the input tensor."""
        # Self-attention with residual connection
        x_norm1 = self.ln1(x)
        attn_output = self.attn(x_norm1, token_positions)
        x = x + attn_output

        # Feed-forward network with residual connection
        x_norm2 = self.ln2(x)
        ff_output = self.ffn(x_norm2)
        x = x + ff_output

        return x