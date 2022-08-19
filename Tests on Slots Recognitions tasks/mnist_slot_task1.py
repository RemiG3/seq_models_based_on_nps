import numpy as np
from mnist_slot_dataset import Generate_MNIST_digit

def get_data():
     return {
        0: lambda data: Generate_MNIST_digit([1, 1, 1, None], data),
        1: lambda data: Generate_MNIST_digit([1, 1, 2, None], data),
        2: lambda data: Generate_MNIST_digit([1, 2, 1, None], data),
        3: lambda data: Generate_MNIST_digit([1, 2, 2, None], data),
        4: lambda data: Generate_MNIST_digit([2, 1, 1, None], data),
        5: lambda data: Generate_MNIST_digit([2, 1, 2, None], data),
        6: lambda data: Generate_MNIST_digit([2, 2, 1, None], data),
        7: lambda data: Generate_MNIST_digit([2, 2, 2, None], data),
    }