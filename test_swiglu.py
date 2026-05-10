import torch
from nanofm.modeling.transformer_layers import SwiGLU, Block

B, L, D = 2, 16, 128
x = torch.randn(B, L, D)

ffn = SwiGLU(D, hidden_features=4*D, out_features=D)
y = ffn(x)

print("SwiGLU:", x.shape, "->", y.shape)
assert y.shape == x.shape
assert torch.isfinite(y).all()

block = Block(dim=D, head_dim=32, use_swiglu=True)
y = block(x)

print("Block:", x.shape, "->", y.shape)
assert y.shape == x.shape
assert torch.isfinite(y).all()

print("All tests passed.")