from torch import nn


# class LayerNorm(nn.Module):
#     def __init__(self):
#         super(LayerNorm, self).__init__()
#         self.layernorm = nn.functional.layer_norm

#     def forward(self, x):
#         x = self.layernorm(x, list(x.size()[1:]))
#         return x

# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)

# class UnFlatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), 16, 2, 2)


# def get_encoder(hidden_dim, nb_chans):
#     return nn.Sequential(
#         nn.Conv2d(nb_chans, 16, kernel_size=3),
#         nn.ELU(),
#         LayerNorm(),
#         Flatten(),
#         nn.Linear(64, hidden_dim),
#         nn.ELU(),
#         LayerNorm()
#     )

# def get_decoder(hidden_dim, nb_chans):
#     return nn.Sequential(
#         nn.Linear(hidden_dim, 64),
#         nn.ELU(),
#         LayerNorm(),
#         UnFlatten(),
#         nn.ConvTranspose2d(16, nb_chans, kernel_size=3),
#         nn.Sigmoid()
#     )


class LayerNorm(nn.Module):
    def __init__(self):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.functional.layer_norm

    def forward(self, x):
        x = self.layernorm(x, list(x.size()[1:]))
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 16, 1, 1)


def get_encoder(hidden_dim, nb_chans):
    return nn.Sequential(
        nn.Conv2d(nb_chans, 16, kernel_size=4),
        nn.ELU(),
        LayerNorm(),
        Flatten(),
        nn.Linear(16, hidden_dim),
        nn.ELU(),
        LayerNorm()
    )

def get_decoder(hidden_dim, nb_chans):
    return nn.Sequential(
        nn.Linear(hidden_dim, 16),
        nn.ELU(),
        LayerNorm(),
        UnFlatten(),
        nn.ConvTranspose2d(16, nb_chans, kernel_size=4),
        nn.Sigmoid()
    )
