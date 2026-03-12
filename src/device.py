"""
Centralized device configuration

Import with `from src.device import device` across the project.
"""
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
