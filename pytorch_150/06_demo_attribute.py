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


