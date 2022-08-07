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
        return input.view(input.size(0), 32, 2, 2)


def get_encoder(hidden_dim, nb_chans):
    return nn.Sequential(
        nn.Conv2d(nb_chans, 16, kernel_size=3),
        nn.ELU(),
        LayerNorm(),
        nn.MaxPool2d(kernel_size=(2, 2)),#, stride=2),
        nn.Conv2d(16, 32, kernel_size=3),
        nn.ELU(),
        LayerNorm(),
        nn.MaxPool2d(kernel_size=(2, 2)),#, stride=2),
        Flatten(),
        nn.Linear(128, hidden_dim),
        nn.ELU(),
        LayerNorm()
    
        # Flatten(),
        # nn.Linear(14*14, 64),
        # nn.ELU(),
        # LayerNorm(),
        # nn.Linear(64, HIDDEN_DIM),
        # nn.ELU(),
        # LayerNorm(),
    )

def get_decoder(hidden_dim, nb_chans):
    return nn.Sequential(
        # nn.Sigmoid(),
        # LayerNorm(),
        nn.Linear(hidden_dim, 128),
        # nn.ReLU(),
        nn.ELU(),
        LayerNorm(),
        UnFlatten(),
        #Interpolate(scale_factor=2, mode='bilinear'),
        #nn.ReplicationPad2d(1),
        nn.ConvTranspose2d(32, 16, kernel_size=3),
        nn.Upsample(scale_factor=2, mode='bilinear'),
        # Interpolate(scale_factor=2, mode='bilinear'),
        nn.ReplicationPad2d(2),
        nn.ELU(),
        # nn.ReLU(),
        LayerNorm(),
        nn.ConvTranspose2d(16, nb_chans, kernel_size=3),
        nn.Sigmoid()
    
        # nn.Linear(HIDDEN_DIM, 64),
        # nn.ELU(),
        # LayerNorm(),
        # nn.Linear(64, 14*14),
        # UnFlattenDense(),
        # nn.Sigmoid(),
    )

