class Patches_callback:
    def __init__(self, size, stride):
        self.__name__ = f'Paches_callback(size={size},stride={stride})'
        self.size = size
        self.stride = stride
    
    def __call__(self, x, encoder=None):
        n_chans = x.size(1)
        return x.unfold(-2, self.size, self.stride).unfold(-2, self.size, self.stride).permute(0,2,3,1,4,5).reshape((-1, n_chans, self.size, self.size))


class Patches_encoded_callback:
    def __init__(self, size, stride):
        self.__name__ = f'Patches_encoded_callback(size={size},stride={stride})'
        self.size = size
        self.stride = stride
    
    def __call__(self, x, encoder=None):
        n_chans = x.size(1)
        patches = x.unfold(-2, self.size, self.stride).unfold(-2, self.size, self.stride).permute(0,2,3,1,4,5).reshape((-1, n_chans, self.size, self.size))
        if encoder is None: # In case of preprocessing for the autoencoder (if any)
            return patches
        else:
            patches_encoded = encoder(patches)
            return patches_encoded.reshape((x.size(0), -1, patches_encoded.size(-1)))
