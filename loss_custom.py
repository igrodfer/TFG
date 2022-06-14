from bz2 import compress
import torch
import torch.nn as nn

def amplitude_reduced_MSE(output,compressed,target):
    loss = torch.mean((output-target)**2)
    amplitude = compressed**2
    return loss + amplitude