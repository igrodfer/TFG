from bz2 import compress
import torch
import torch.nn as nn

def get_amplitude_reduced_MSE(l=0.1):
    def amplitude_reduced_MSE(output,compressed,target):
        loss = torch.mean((output-target)**2)
        amplitude = torch.mean(compressed**2)
        return loss + l * amplitude
    return amplitude_reduced_MSE

def custom_MSE(output,compressed,target):
    loss = torch.mean((output-target)**2)
    return loss