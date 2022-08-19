from torch import nn


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
        return input.view(input.size(0), 32, 1, 1)


def get_encoder(hidden_dim, nb_chans):
    return nn.Sequential(
        nn.Conv2d(nb_chans, 16, kernel_size=(3, 3)),
        nn.ELU(),
        LayerNorm(),
        nn.MaxPool2d(kernel_size=(4, 4)),
        nn.Conv2d(16, 32, kernel_size=(5, 5)),
        nn.ELU(),
        LayerNorm(),
        nn.MaxPool2d(kernel_size=(2, 2)),
        Flatten(),
        nn.Linear(32, hidden_dim, bias=True),
        nn.ELU(),
        LayerNorm(),
    )

def get_decoder(hidden_dim, nb_chans):
    return nn.Sequential(
        nn.Linear(hidden_dim, 32),
        nn.ELU(),
        LayerNorm(),
        UnFlatten(),
        nn.ConvTranspose2d(32, 16, kernel_size=(5, 5)),
        nn.Upsample(scale_factor=4, mode='bilinear'),
        nn.ReplicationPad2d(3),
        nn.ELU(),
        LayerNorm(),
        nn.ConvTranspose2d(16, nb_chans, kernel_size=(3, 3)),
        nn.Sigmoid()
    )

