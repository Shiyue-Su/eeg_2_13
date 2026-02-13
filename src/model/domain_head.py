import torch
import torch.nn as nn


class DomainHead(nn.Module):
    def __init__(self, out_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)
