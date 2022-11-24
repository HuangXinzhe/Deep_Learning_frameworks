import torch

"""
torch.where
利用条件判断输出结果

用途：
1、利用阈值来对tensor进行二值化
2、只想计算大于某个值的loss，可以将小于某个值的loss定义为0
"""

# a = torch.rand(4, 4)
# b = torch.rand(4, 4)
#
# print(a)
# print(b)
#
# # 当满足条件时，输出a中元素，当不满足条件时输出条件b
# out = torch.where(a > 0.5, a, b)
# print(out)

"""
torch.index_select
在指定维度上选择数据元素
"""

# a = torch.rand(4, 4)
# print(a)
# out = torch.index_select(a,
#                          dim=0,
#                          index=torch.tensor([0, 3, 2]))
# print(out, out.shape)

"""
torch.gather

dim=0, out[i, j, k] = input[indec[i, j, k], j, k]
dim=1, out[i, j, k] = input[i, indec[i, j, k], k]
dim=2, out[i, j, k] = input[i, j, indec[i, j, k]]
"""

# a = torch.linspace(1, 16, 16).view(4, 4)
# print(a)
# out = torch.gather(a,
#                    dim=0,
#                    index=torch.tensor([[0, 1, 1, 1],
#                                        [0, 1, 2, 2],
#                                        [0, 1, 3, 3]]))
#
# print(out, out.shape)

"""
torch.masked_select
想要选择其中的某一些值
"""

# a = torch.linspace(1, 16, 16).view(4, 4)
# print(a)
#
# mask = torch.gt(a, 8)
# print(mask)
#
# out = torch.masked_select(a, mask)
# print(out)

"""
torch.take
将输出拉成一个向量
"""

# a = torch.linspace(1, 16, 16).view(4, 4)
# print(a)
#
# b = torch.take(a,
#                index=torch.tensor([0, 15, 13, 10]))
# print(b)

"""
torch.nanzero
输出非0元素的坐标
tensor的稀疏表示
"""

a = torch.tensor([[0, 1, 2, 0], [2, 3, 0, 1]])
print(a)

out = torch.nonzero(a)
print(out)

