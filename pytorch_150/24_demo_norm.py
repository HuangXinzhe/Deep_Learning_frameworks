import torch

a = torch.rand(1, 1)
b = torch.rand(1, 1)

print(a, b)

print(torch.dist(a, b, p=1))
print(torch.dist(a, b, p=2))
print(torch.dist(a, b, p=3))

print(torch.norm(a))
print(torch.norm(a, p=1))
print(torch.norm(a, p='fro'))  # 核范数