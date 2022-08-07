from torch import nn


class LayerNorm(nn.Module):
    def __init__(self):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.functional.layer_norm

    def forward(self, x):
        x = self.layernorm(x, list(x.size()[1:]))
        return x

def get_encoder(hidden_dim, n_slots, device):
    return nn.ModuleList([nn.Sequential(
        nn.Linear(1, hidden_dim),
        nn.ELU(),
        LayerNorm()
    ).to(device) for _ in range(n_slots)])

def get_decoder(hidden_dim, n_slots, device):
    return nn.ModuleList([nn.Sequential(
        nn.Linear(hidden_dim, 1),
        nn.ELU(),
        LayerNorm()
    ).to(device) for _ in range(n_slots)])

