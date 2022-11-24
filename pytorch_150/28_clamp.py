import torch

a = torch.rand(2, 2) * 10
print(a)

a = a.clamp(1, 2)
print(a)

