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


############### Hyperparameters parsing ################

def parse_boolean(value):
    value = value.lower()
    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False
    return False

def parse_str_with_None(value):
    value = str(value).replace("'", '').replace('"', '')
    return None if value == 'None' else value

def parse_int_with_none(value):
    value = str(value).replace("'", '').replace('"', '')
    return None if ((value == 'None') or (int(value) <= 0)) else int(value)

def get_args(str_args, dic_type=None):
    if (str_args == '') or (str_args == "''") or (str_args == 'None'):
        return []
    sep = '=' if('=' in str_args) else ':' if (':' in str_args) else None
    if sep:
        args = dict(map(lambda e: map(str, e.split(sep)), str_args.split(',')))
        return args if dic_type is None else {k: dic_type[k](args[k]) for k in args}
    else:
        args = list(map(str, str_args.split(',')))
        return args if dic_type is None else [t(e) for t, e in zip(dic_type.values(), args)]

def get_strat(param):
        if param is not None:
            split = param.split('(')
            strat = split[0]
            start, stop, step = list(map(float, split[1][:-1].split(',')))
            if strat == 'lin_dec':
                return Tau_linear_decreasing_strategy(start, stop, step)
            elif strat == 'exp_dec':
                return Tau_exponential_decreasing_strategy(start, stop, step)
        return None