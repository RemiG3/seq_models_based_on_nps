import torch
import time
import random
import numpy as np


def save_handler(save_name, extension='.pickle'):
    save_name = save_name if save_name.endswith(extension) else save_name+extension
    return save_name

def unset_seed():
    set_seed(time.time())

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(int(seed))
    torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.backends.cudnn.deterministic = True


############### Tau strategies ################

class Tau_linear_decreasing_strategy:
    def __init__(self, start, end, step):
        self.__name__ = f'Tau_linear_decreasing_strategy(start={start},end={end},step={step})'
        self.cur = start
        self.end = end
        self.step = step

    def __call__(self, model, attribute_name):
        setattr(model, attribute_name, self.cur)
        if self.cur > self.end:
            self.cur = max(self.end, self.cur-self.step)

class Tau_exponential_decreasing_strategy:
    def __init__(self, start, end, decay):
        self.__name__ = f'Tau_exponential_decreasing_strategy(start={start},end={end},decay={decay})'
        self.cur = start
        self.end = end
        self.decay = decay
        self.i = 0

    def __call__(self, model, attribute_name):
        setattr(model, attribute_name, self.cur)
        if self.cur > self.end:
            self.cur = max(self.end, self.end + (self.cur-self.end)*np.exp(-self.decay*self.i))
            self.i += 1