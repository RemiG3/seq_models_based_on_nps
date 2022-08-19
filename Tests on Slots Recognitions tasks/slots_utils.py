
class Slots_encoded_callback:
    def __init__(self, s_dim):
        self.__name__ = f'Slots_encoded_callback(s_dim={s_dim})'
        self.s_dim = s_dim
    
    def __call__(self, x, encoder=None):
        if encoder is None:
            return x.reshape(-1, x.size(-3), x.size(-2), x.size(-1))
        else:
            return encoder(x.reshape(-1, x.size(-2), x.size(-1)).unsqueeze(1)).unsqueeze(1).reshape(x.size(0), -1, self.s_dim)
