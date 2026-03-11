"""
デバイス設定の一元管理
Centralized device configuration

プロジェクト全体で `from src.device import device` としてインポートして使う。
Import with `from src.device import device` across the project.
"""
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
