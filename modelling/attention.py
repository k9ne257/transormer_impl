# modelling/attention.py
import math
import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    Scaled dot-product attention with:
      - padding mask (provided as binary attention_mask, shape (B, Tk))
      - optional future/causal mask (computed internally) when mask_future=True

    Forward signature matches the test:
      attention_layer(query, key, value, attention_mask) -> (B, Tq, D)
    """

    def __init__(self, mask_future: bool = False):
        super().__init__()
        self.mask_future = mask_future

    def forward(
        self,
        query: torch.Tensor,          # (B, Tq, D)
        key: torch.Tensor,            # (B, Tk, D)
        value: torch.Tensor,          # (B, Tk, D)
        attention_mask: torch.Tensor  # (B, Tk) with 1=keep, 0=mask
    ) -> torch.Tensor:
        B, Tq, D = query.shape
        Tk = key.shape[1]

        # (B, Tq, Tk) scaled dot-product scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(D)

        # 1) Padding mask over KEYS: mask out positions where attention_mask == 0
        # attention_mask: (B, Tk) -> (B, 1, Tk) broadcast over Tq
        key_mask = attention_mask.to(dtype=torch.bool).unsqueeze(1)  # True = keep
        scores = scores.masked_fill(~key_mask, float("-inf"))

        # 2) Future/causal mask (only valid/needed when doing self-attention)
        # In your tests, mask_future is used with query==key==value, so Tq==Tk.
        if self.mask_future:
            # mask upper triangle (j > i): block attending to future keys
            # shape: (Tq, Tk)
            future_mask = torch.triu(
                torch.ones((Tq, Tk), device=scores.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(future_mask.unsqueeze(0), float("-inf"))

        # (B, Tq, Tk)
        attn = torch.softmax(scores, dim=-1)

        # (B, Tq, D)
        out = torch.matmul(attn, value)
        return out
