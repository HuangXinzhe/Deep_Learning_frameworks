import torch

a = torch.rand(2, 2)

print(a)
print(torch.mean(a, dim=0))
print(torch.sum(a, dim=0))
print(torch.prod(a, dim=0))

print(torch.argmax(a, dim=0))
print(torch.argmin(a, dim=0))

print(torch.std(a))
print(torch.var(a))

print(torch.median(a))
print(torch.mode(a))

b = torch.rand(2, 2)
print(b)
print(torch.histc(b, 6, 0, 0))  # 当不设置最大最小值时，将数据中的最大值与最小值作为最大最小值

c = torch.randint(0, 10, [10])
print(c)
print(torch.bincount(c))  # 统计频率，只能处理一维数据

# 统计某一类别样本的个数
