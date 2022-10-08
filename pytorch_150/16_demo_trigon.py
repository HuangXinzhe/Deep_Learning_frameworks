import torch

"""
三角函数
有原地操作
"""
a = torch.rand(2, 3)
b = torch.cos(a)
print(a)
print(b)