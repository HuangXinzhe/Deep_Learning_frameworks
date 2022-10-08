import torch

"""
进行广播运算时，以最右侧的参数为准进行判断，
当有一个为1时，另一个均可，否则两者需相等或成倍数
"""
a = torch.rand(2, 3)
b = torch.rand(3)
c = a + b
print(a)
print(b)
print(c)
print(c.shape)