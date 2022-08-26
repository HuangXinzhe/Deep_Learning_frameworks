import torch

# 直接定义数据
a = torch.Tensor([[1, 2], [3, 4]])
print(a)
print(a.type())

# 定义数据形状，值没有定义随机
a = torch.Tensor(2, 3)
print(a)
print(a.type())

"""几种特殊的tensor"""
# 全1tensor
a = torch.ones(2, 2)
print(a)
print(a.type())

# 对角线全1tensor
a = torch.eye(2, 2)
print(a)
print(a.type())

# 全0tensor
a = torch.zeros(2, 2)
print(a)
print(a.type())

# 定义相同形状不同值的tensor
b = torch.Tensor(2, 3)
b = torch.zeros_like(b)
b = torch.ones_like(b)
print(b)
print(b.type())

"""随机"""
# 0到1之间的随机值
a = torch.rand(2, 2)
print(a)
print(a.type())

# 随机并满足某种分布
# 正太分布
a = torch.normal(mean=0.0, std=torch.rand(5))
print(a)
print(a.type())

a = torch.normal(mean=torch.rand(5), std=torch.rand(5))
print(a)
print(a.type())

# 均匀分布，需指定形状大小
# -1到1之间的均匀分布
a = torch.Tensor(2, 2).uniform_(-1, 1)
print(a)
print(a.type())

"""序列"""
# 从0到10，步长为1的序列，包前不包后
a = torch.arange(0, 10, 1)
print(a)
print(a.type())

# 等间隔序列
a = torch.linspace(2, 10, 3)
print(a)
print(a.type())

# 打乱序列
a = torch.randperm(10)
print(a)
print(a.type())


##########################################
import numpy as np
a = np.array([[1, 2], [2, 3]])
print(a)






