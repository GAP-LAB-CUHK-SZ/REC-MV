import numpy as np
import torch

def faster_matrix_power(matrix, power = 1):
    assert power != 0
    if power == 1:
        return matrix
    elif power & 1:
        return torch.sparse.mm(faster_matrix_power(matrix, power-1), matrix)
    else:
        tmp = faster_matrix_power(matrix, power//2)
        return  torch.sparse.mm(tmp, tmp)


