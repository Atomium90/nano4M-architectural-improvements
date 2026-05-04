# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# Some functions are based on the timm and 4M code bases
# https://github.com/huggingface/pytorch-image-models
# https://github.com/apple/ml-4m
# --------------------------------------------------------

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def rotate_half(x):
    # splits last dim in half and swaps signs
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k):
    """
    q, k: [B, H, L, D]
    applies rotary embedding on last dim
    """
    B, H, L, D = q.shape
    device = q.device

    half_dim = D // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))

    positions = torch.arange(L, device=device).float()
    freqs = torch.einsum("i,j->ij", positions, inv_freq)  # [L, D/2]

    emb = torch.cat([freqs, freqs], dim=-1)  # [L, D]
    emb = emb[None, None, :, :]  # [1,1,L,D]

    cos = emb.cos()
    sin = emb.sin()

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot

def build_alibi_bias(num_heads: int, seq_len: int, device):
    """
    Simple ALiBi (no fancy masking logic, works with full + masked attention).
    Produces shape: [1, num_heads, seq_len, seq_len]
    """
    def get_slopes(n):
        # classic ALiBi slope construction (simplified stable version)
        def get_pow2_slopes(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        # fallback safe slopes
        return get_pow2_slopes(n)

    slopes = torch.tensor(get_slopes(num_heads), device=device)
    arange = torch.arange(seq_len, device=device)

    # distance matrix [L, L]
    dist = arange[None, :] - arange[:, None]  # (query - key)
    dist = -torch.abs(dist).unsqueeze(0).unsqueeze(0)  # [1,1,L,L]

    bias = slopes[:, None, None] * dist  # [H, L, L]
    
class LayerNorm(nn.Module):
    """Custom implementation of LayerNorm with the option to disable the bias term."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_buffer("bias", torch.zeros(normalized_shape))

        # Normalized shape must be a tuple for F.layer_norm
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, eps=self.eps)


class Mlp(nn.Module):
    """
    MLP module with GELU activation.

    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features (optional)
        out_features: Number of output features (optional)
        bias: Whether to include bias in the linear layers
    """
    def __init__(self, 
            in_features: int, 
            hidden_features: Optional[int] = None, 
            out_features: Optional[int] = None, 
            bias: bool = False,
        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.l1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.activation = nn.GELU()
        self.l2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention module.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        qkv_bias: Whether to include bias in the QKV linear layers
        proj_bias: Whether to include bias in the attention output projection
    """
    def __init__(self, dim: int, head_dim: int = 64, qkv_bias: bool = False, proj_bias: bool = False):
        super().__init__()
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        # Define here the linear layer(s) producing K, Q, V from the input x
        # Hint: Do you need to define three different projections, or can you use a single one for all three?
        self.QKV_layer = nn.Linear(dim, 3*dim, bias=qkv_bias)

        self.attn_out_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.pos_encoding = "none"
        self.max_seq_len = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape # Batch size, sequence length, and dimension

        # Compute the keys K, queries Q, and values V from x. Each should be of shape [B num_heads L head_dim].
        qkv = self.QKV_layer(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "b l (nh hd) -> b nh l hd", nh=self.num_heads)
        k = rearrange(k, "b l (nh hd) -> b nh l hd", nh=self.num_heads)

        if self.pos_encoding == "rope":
            q, k = apply_rope(q, k)

        v = rearrange(v, "b l (nh hd) -> b nh l hd", nh=self.num_heads)
       
        # Compute the attention matrix (pre softmax) and scale it by 1/sqrt(d_k). It should be of shape [B num_heads L L].
        # Hint: Use the already defined self.scale


        attn = q @ k.transpose(-1, -2)
        attn = attn * self.scale

        if self.pos_encoding == "alibi":
            B, H, L, _ = attn.shape
            bias = build_alibi_bias(H, L, attn.device)
            attn = attn + bias

        if mask is not None:
            mask = rearrange(mask, "b n m -> b 1 n m") # Unsqueeze for multi-head attention
            # Apply the optional attention mask. Wherever the mask is False, replace the attention 
            # matrix value by negative infinity → zero attention weight after softmax.
            attn = attn.masked_fill(~mask, float('-inf'))

        # Compute the softmax over the last dimension
        attn = attn.softmax(dim=-1)

        # Weight the values V by the attention matrix and concatenate the different attention heads
        # Make sure to reshape the output to the original shape of x, i.e. [B L D]
        x = attn @ v
        x = rearrange(x, "b nh l hd -> b l (nh hd)")

        # Output projection
        x = self.attn_out_proj(x)
        return x

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention module.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        qkv_bias: Whether to include bias in the QKV linear layers
        proj_bias: Whether to include bias in the attention output projection
    """
    def __init__(self, dim: int, head_dim: int = 64, qkv_bias: bool = False, proj_bias: bool = False):
        super().__init__()
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        # Define here the linear layer producing Q from the input x
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # Define here the linear layers producing K, V from the context
        # Hint: Do you need to define two different projections, or can you use a single one for both?
        self.kv_proj = nn.Linear(dim, 2*dim, bias=qkv_bias)

        self.attn_out_proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape # Batch size, x sequence length (N), and dimension
        _, M, _ = context.shape # _, context sequence length (M), _

        # Compute the queries Q from x. It should be of shape [B num_heads N head_dim].
        q = self.q_proj(x)
        q = rearrange(q, "b n (nh hd) -> b nh n hd", nh = self.num_heads)

        # Compute the keys K and values V from the context. Each should be of shape [B num_heads M head_dim].
        kv = self.kv_proj(context)
        k, v = kv.chunk(2, dim=-1)
        k = rearrange(k, "b m (nh hd) -> b nh m hd", nh = self.num_heads)
        v = rearrange(v, "b m (nh hd) -> b nh m hd", nh = self.num_heads)

        # Compute the attention matrix (pre softmax) and scale it by 1/sqrt(d_k). It should be of shape [B num_heads N M].
        # Hint: Use the already defined self.scale
        attn = q @ k.transpose(-1, -2)
        attn = attn * self.scale

        if mask is not None:
            mask = rearrange(mask, "b n m -> b 1 n m") # Unsqueeze for multi-head attention
            # Apply the optional attention mask. Wherever the mask is False, replace the attention 
            # matrix value by negative infinity → zero attention weight after softmax.
            attn = attn.masked_fill(~mask, float('-inf'))

        # Compute the softmax over the last dimension
        attn = attn.softmax(dim=-1)

        # Weight the values V by the attention matrix and concatenate the different attention heads
        # Make sure to reshape the output to the original shape of x, i.e. [B N D]
        x = attn @ v
        x = rearrange(x, "b nh n hd -> b n (nh hd)")
        
        # Output projection
        x = self.attn_out_proj(x)

        return x


class Block(nn.Module):
    """
    Basic transformer block with a multi-head self-attention mechanism and a feed-forward MLP.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
    """
    def __init__(self, dim: int, head_dim: int = 64, mlp_ratio: float = 4., use_bias: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(dim, bias=use_bias)
        self.attn = Attention(dim, head_dim=head_dim, qkv_bias=use_bias, proj_bias=use_bias)
        self.norm2 = LayerNorm(dim, bias=use_bias)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Self-attention pass
        x_attn = self.attn(self.norm1(x), mask=mask)
        x = x + x_attn

        # MLP pass
        x_mlp = self.mlp(self.norm2(x))
        x = x + x_mlp

        return x

class DecoderBlock(nn.Module):
    """
    Basic transformer decoder block with a multi-head self-attention, 
    a multi-head cross-attention, and a feed-forward MLP layer.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
    """
    def __init__(self, dim: int, head_dim: int = 64, mlp_ratio: float = 4., use_bias: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(dim, bias=use_bias)
        self.query_norm = LayerNorm(dim, bias=use_bias)
        self.context_norm = LayerNorm(dim, bias=use_bias)
        self.norm2 = LayerNorm(dim, bias=use_bias)

        self.self_attn = Attention(dim, head_dim=head_dim, qkv_bias=use_bias, proj_bias=use_bias)
        self.cross_attn = CrossAttention(dim, head_dim=head_dim, qkv_bias=use_bias, proj_bias=use_bias)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, bias=use_bias)

    def forward(self, 
            x: torch.Tensor, 
            context: torch.Tensor, 
            sa_mask: Optional[torch.Tensor] = None, # Self-attention mask
            xa_mask: Optional[torch.Tensor] = None, # Cross-attention mask
        ) -> torch.Tensor:

        # Self-attention, then cross-attention, then MLP
        # Make sure to apply the self-attention mask (sa_mask) to the self-attention layer,
        # and the cross-attention mask (xa_mask) to the cross-attention layer.
        # Don't forget to add the residual connections after each layer, and
        # to apply the normalizations on the inputs of each layer.
        
        # Self-attention pass
        x = x + self.self_attn(self.norm1(x), mask=sa_mask)

        # Cross-attention pass
        x = x + self.cross_attn(self.query_norm(x), self.context_norm(context), mask=xa_mask)

        # MLP pass
        x = x + self.mlp(self.norm2(x))

        return x


class TransformerTrunk(nn.Module):
    """Basic Transformer trunk definition that can be used for encoder-only,
    decoder-only and prefixLM models, depending on the attention mask applied.

    Args:
        dim: Transformer dimension
        depth: Number of transformer layers
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
    """
    def __init__(
        self,
            dim: int = 512,
            depth: int = 8,
            head_dim: int = 64,
            mlp_ratio: float = 4.0,
            use_bias: bool = False,
        ):
        super().__init__()

        # Create a list of transformer blocks and wrap inside nn.ModuleList
        self.blocks = nn.ModuleList([
            Block(dim=dim, head_dim=head_dim, mlp_ratio=mlp_ratio, use_bias=use_bias) 
            for _ in range(depth)
        ])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Forward pass through every individual block
        for block in self.blocks:
            x = block(x, mask=mask)

        return x

class TransformerDecoderTrunk(nn.Module):
    """Basic Transformer decoder with interleaved self- and cross-attention, that can
    be used as the decoder for encoder-decoder models.

    Args:
        dim: Transformer dimension
        depth: Number of transformer layers
        head_dim: Dimension of each attention head
        mlp_ratio: Ratio of MLP hidden dimension to transformer dimension
        use_bias: Whether to include bias in the QKV, attention output projection and MLP layers
    """
    def __init__(
        self,
            dim: int = 512,
            depth: int = 8,
            head_dim: int = 64,
            mlp_ratio: float = 4.0,
            use_bias: bool = False,
        ):
        super().__init__()

        # Create a list of transformer decoder blocks and wrap inside nn.ModuleList
        self.blocks = nn.ModuleList([
            DecoderBlock(dim=dim, head_dim=head_dim, mlp_ratio=mlp_ratio, use_bias=use_bias)
            for _ in range(depth)
        ])
    
    def forward(
            self, 
            x: torch.Tensor, 
            context: torch.Tensor, 
            sa_mask: Optional[torch.Tensor] = None, # Self-attention mask
            xa_mask: Optional[torch.Tensor] = None, # Cross-attention mask
        ) -> torch.Tensor:
        
        # Forward pass through every individual block
        for block in self.blocks:
            x = block(x, context, sa_mask=sa_mask, xa_mask=xa_mask)

        return x