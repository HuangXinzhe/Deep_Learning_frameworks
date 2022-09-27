import torch

# dev = torch.device("cpu")
# dev = torch.device("cuda:0")
dev = torch.device("cuda")  # 使用默认显卡
a = torch.tensor([2, 2],
                 dtype=torch.float32,
                 device=dev)
print(a)

# 稀疏的张量
# 坐标
i = torch.tensor([[0, 1, 2], [0, 1, 2]])
# 值
v = torch.tensor([1, 2, 3])
a = torch.sparse_coo_tensor(i, v, (4, 4))
print(a)

# 稀疏的张量转为稠密的张量
a = torch.sparse_coo_tensor(i, v, (4, 4)).to_dense()
print(a)

# 定义数据类型和GPU
a = torch.sparse_coo_tensor(i, v, (4, 4),
                            dtype=torch.float32,
                            device=dev).to_dense()
print(a)

"""
数据分别放在CPU或GPU上的理由
CPU：数据分配，数据读取，数据预处理操作
GPU：参数计算，推理，反向传播

通过对资源的合理分配来实现资源利用率的最大化，使得网络训练迭代更快
"""

