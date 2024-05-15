import numpy as np


def convert_to_binary_with_floor(X):

    floored_X = np.floor(X).astype(int)

    binary_string = ''.join(format(int(x), '08b') for x in floored_X)

    return binary_string


