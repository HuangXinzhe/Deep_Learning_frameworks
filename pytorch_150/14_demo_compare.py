import torch

"""
compare
"""
print("====compare====")
a = torch.rand(2, 3)
b = torch.rand(2, 3)

print(a)
print(b)


print(torch.eq(a, b))
print(torch.equal(a, b))
print(torch.ge(a, b))
print(torch.gt(a, b))
print(torch.le(a, b))
print(torch.lt(a, b))
print(torch.ne(a, b))

"""
sort
"""
print("====sort====")
a = torch.tensor([[1, 4, 4, 3, 5],
                  [2, 3, 1, 3, 5]])

print(a.shape)
# dim表示在哪一个维度上比较数值
# descending为True时为降序排序，False为升序排序，默认为升序排序
print(torch.sort(a,
                 dim=1,
                 descending=True))

"""
topk
"""
print("====topk====")
a = torch.tensor([[2, 4 , 3 , 1, 5],
                  [2, 3, 5, 1, 4]])
print(a.shape)

print(torch.topk(a, k=2, dim=0))

"""
kthvalue
"""
print("====kthvalue====")
print(torch.kthvalue(a, k=2, dim=0))
print(torch.kthvalue(a, k=2, dim=0))

"""
有界、无界和nan
"""
print("====有界、无界和nan====")
a = torch.rand([2, 3])
print(a)
print(a/0)
print(torch.isfinite(a))
print(torch.isfinite(a/0))
print(torch.isinf(a/0))
print(torch.isnan(a))

import numpy as np
a = torch.tensor([1, 2, np.nan])
print(torch.isnan(a))