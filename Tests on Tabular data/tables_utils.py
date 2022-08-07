import torch

class Transform_encoded_callback:
    def __init__(self):
        self.__name__ = f'Transform_encoded_callback()'
    
    def __call__(self, x, encoder=None):
        if type(encoder) == torch.nn.modules.container.ModuleList:
            encoding = []
            for n in range(x.size(-1)):
                encoding.append(encoder[n](x[:,n].unsqueeze(-1)).unsqueeze(1))
            return torch.cat(tuple(encoding), dim=1)
        else:
            features = x.unsqueeze(-1) # (bs, n_slots, 1)
            if encoder is None: # In case of preprocessing for the autoencoder (if any)
                return features
            else:
                return encoder(features)
