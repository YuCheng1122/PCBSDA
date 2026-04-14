"""
MalConv: Malware Detection by Eating Executables
Anderson et al., 2018 (AAAI Workshop on Artificial Intelligence for Cyber Security)

Architecture:
  Embedding(256+1 tokens, embed_dim)
  → Conv1d (gated, filter_size, num_filters)  -- two parallel conv for gate & signal
  → Global Max Pooling
  → FC → num_classes

Memory note:
  Processing a full 2MB sequence with Conv1d creates a huge intermediate activation
  tensor (B * embed_dim * T * 4 bytes). To stay within GPU memory on an 11GB card,
  forward() processes the sequence in chunks of `chunk_size` bytes and accumulates
  the running max, so peak activation memory scales with chunk_size rather than T.
  This is mathematically equivalent to global max pooling over the full sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MalConv(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 8,
        num_filters: int = 128,
        filter_size: int = 512,
        stride: int = 512,
        dropout: float = 0.5,
        chunk_size: int = 65536,   # process this many bytes at a time during forward
    ):
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.chunk_size = chunk_size

        # vocab: 0 = pad, 1-256 = byte values
        self.embed = nn.Embedding(257, embed_dim, padding_idx=0)
        # gated convolution: two parallel conv ops
        self.conv_signal = nn.Conv1d(embed_dim, num_filters, filter_size, stride=stride, bias=True)
        self.conv_gate   = nn.Conv1d(embed_dim, num_filters, filter_size, stride=stride, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters, num_filters)
        self.fc2 = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        # x: (B, T) long
        T = x.shape[1]
        # overlap between chunks must be (filter_size - 1) so conv windows
        # that span a chunk boundary are still evaluated.
        overlap = self.filter_size - 1
        step    = self.chunk_size - overlap

        running_max = None  # (B, num_filters)

        start = 0
        while start < T:
            end  = min(start + self.chunk_size, T)
            chunk = x[:, start:end]                         # (B, chunk_len)

            emb = self.embed(chunk)                         # (B, chunk_len, E)
            emb = emb.permute(0, 2, 1)                      # (B, E, chunk_len)

            if emb.shape[2] < self.filter_size:
                # chunk is shorter than one filter window — skip
                start += step
                continue

            signal = self.conv_signal(emb)                  # (B, F, L)
            gate   = torch.sigmoid(self.conv_gate(emb))     # (B, F, L)
            gated  = signal * gate                          # (B, F, L)

            chunk_max, _ = gated.max(dim=2)                 # (B, F)

            if running_max is None:
                running_max = chunk_max
            else:
                running_max = torch.max(running_max, chunk_max)

            start += step

        if running_max is None:
            # sequence shorter than filter_size — return zeros
            running_max = x.new_zeros(x.shape[0], self.conv_signal.out_channels,
                                      dtype=torch.float)

        pooled = self.dropout(running_max)
        out    = F.relu(self.fc1(pooled))
        out    = self.dropout(out)
        return self.fc2(out)                                # (B, num_classes) logits
