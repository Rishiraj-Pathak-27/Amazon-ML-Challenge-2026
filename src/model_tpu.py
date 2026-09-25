"""
Deep Neural Entity Matcher (PyTorch & PyTorch-XLA for TPU acceleration).
Optimized for high-throughput pairwise matching on Google Colab TPUs.
"""
import torch
import torch.nn as nn


def get_device():
    """Detects TPU (via PyTorch XLA), GPU (CUDA/MPS), or CPU."""
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        return device, "tpu"
    except Exception:
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.drop1(self.act1(self.norm1(self.fc1(x))))
        out = self.drop2(self.act2(self.norm2(self.fc2(out))))
        return residual + out


class DeepEntityMatcher(nn.Module):
    """
    Deep Residual Entity Matcher for pairwise feature scoring.
    Inputs: dense pairwise similarity features (e.g. Levenshtein, Jaccard, token ratios).
    Outputs: match logits / probabilities.
    """
    def __init__(self, in_features=15, hidden_dim=128, num_blocks=3, dropout=0.15):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)  # returns logits
