import torch
import torch.optim as optim

from nanofm.models.fourmUpgraded import FourMUpgraded

torch.manual_seed(0)

model = FourMUpgraded(
    enc_tokens_read_key="enc_tokens",
    dec_tokens_read_key="dec_tokens",
    enc_modalities_read_key="enc_mods",
    dec_modalities_read_key="dec_mods",
    enc_positions_read_key="enc_pos",
    dec_positions_read_key="dec_pos",
    enc_pad_mask_read_key="enc_mask",
    dec_pad_mask_read_key="dec_mask",
    modalities=["rgb"],
    vocab_sizes=[1024],
    max_seq_lens=[64],
    dim=128,
    enc_depth=2,
    dec_depth=2,
    head_dim=32,
    pos_encoding="alibi",
)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)

B = 2
N = 16
M = 8

batch = {
    "enc_tokens": torch.randint(0, 1024, (B, N)),
    "dec_tokens": torch.randint(0, 1024, (B, M)),
    "enc_mods": torch.zeros(B, N, dtype=torch.long),
    "dec_mods": torch.zeros(B, M, dtype=torch.long),
    "enc_pos": torch.arange(N).unsqueeze(0).repeat(B, 1),
    "dec_pos": torch.arange(M).unsqueeze(0).repeat(B, 1),
    "enc_mask": torch.ones(B, N, dtype=torch.bool),
    "dec_mask": torch.ones(B, M, dtype=torch.bool),
}

for step in range(50):

    optimizer.zero_grad()

    loss, _ = model(batch)

    loss.backward()

    optimizer.step()

    if step % 5 == 0:
        print(step, loss.item())
        print("POS:", model.pos_encoding)
