# creating a new DNN as per the paper

import torch
import torch.nn as nn

# softplus activation function
# makes the output non-negative and twice differentiable fulfilling first two requirements
def softplus(x):
    return torch.log(1 + torch.exp(x))

# class Losses:
#     def __init__()