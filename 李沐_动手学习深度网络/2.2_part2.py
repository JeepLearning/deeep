import torch
import numpy as np

"""2.4  运算内存开销"""
# x = torch.tensor([1, 2])
# y = torch.tensor([3, 4])
# id_before = id(y)
# y = y + x
# print(id(y) == id_before)
# 这里输出了False，说明指向了新的内存。

# 如果指定结果到原来的内存，我们可以使用前面介绍的索引来进行替换操作
# x = torch.tensor([1, 2])
# y = torch.tensor([3, 4])
# id_before = id(y)
# y[:] = y + x
# print(id(y) == id_before)
# 这里输出了True，我们把 x + y 的结果写进 y 的内存

# 当然我们也可以使用add函数，或者自加运算符（+=），例如：
# x = torch.tensor([1, 2])
# y = torch.tensor([3, 4])
# id_before = id(y)
# 下述三种都一样
# y += x
# y.add_(x)
# torch.add(x, y, out=y)

# print(id(y) == id_before)

"""2.5  Tensor 同 numpy之间的转化"""
# 我们可以使用numpy()和form_numpy()将两者相互转化，数组会共享相同的内存（因此转化很快），改变其中一个另一个也会发生改变
# 还有一个常用的是将 NumPy 中 array 转化为 Tensor的方法就是torch.tensor()，这会拷贝数据（不再共享数据）会消耗更多的时间和空间

# Tensor 转化为 NumPy
# a = torch.ones(5)
# b = a.numpy()
# print(a, b)

# 证明两者共用同一内存
# a += 1
# print(a, b)
# b += 1
# print(a, b)

# NumPy 转化为 Tensor
# a = np.ones(5)
# b = torch.from_numpy(a)
# print(a, b)

# 证明两者共用同一内存
# a += 1
# print(a, b)
# b += 1
# print(a, b)

# 假如拷贝的话，我们可以利用torch.tensor(),该方法总会进行拷贝，不再进行共享
# a = np.ones(5)
# c = torch.tensor(a)
# a += 1
# print(a, c)

"""2.6  Tensor 在 GPU 上进行计算"""
# x = torch.tensor([1, 2])
# if torch.cuda.is_available():
#     device = torch.device('cuda')               # 这里将 GPU 赋给变量 device
#     y = torch.ones_like(x, device=device)       # 在GPU上创建一个Tensor
#     x = x.to(device)                            # 将 x 转化到 CPU 上
#     z = x + y
#     print(z)
#     print(z.to('cpu', torch.double))            # 这里是将 z 转化到 CPU 上, 同时to()可以改变数据类型
