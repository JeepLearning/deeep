import torch

"""1.1  创建Tensor"""
# 1. 创建一个未初始化的Tensor:
# x = torch.empty(5, 3)
# print(x)

# 2. 创建一个随机初始化的Tensor：
# x = torch.rand(5, 3)
# print(x)

# 3. 创建全为零的Tensor
# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)

# 4. 根据数据直接转化为Tensor
# x = torch.tensor([5.5, 3])
# print(x)

# 5. 可以通过已经创建的Tensor来创建新的Tensor，但是会改变数据类型（当然可以通过自定义）
# x = torch.tensor([4.3, 6])

# 默认情况
# y = x.new_ones(5, 3)
# print("x.dtype:%s, x.device:%s" % (x.dtype, x.device))
# print("y.dtype:%s, y.device:%s" % (y.dtype, y.device))

# 指定情况
# z = torch.randn_like(x, dtype=torch.float64)
# print("z.dtype:%s, z.device:%s" % (z.dtype, z.device))

# 上述的结果，不是默认的会标注
# print(x)
# print(y)
# print(z)

# 查看属性，除了上述中我们的已经提到的属性，还有形状(我们以 x 为例):
# print(x.size())
# print(x.shape)

"""1.2  操作"""
"""1. 算数操作"""
# x = torch.ones(5, 3, dtype=torch.float64)
# y = torch.rand(5, 3)
# 加法形式一
# print(x+y)

# 加法形式二
# print(torch.add(x, y))

# 还可以指定输出(类型可能会发生改变)
# result = torch.empty(5, 3)
# torch.add(x, y, out=result)
# print(result)

# 加法形式三, 这里相当于把 x add to y
# y.add_(x)
# print(y)

"""2. 索引"""
# y = x[0, :]
# y += 1
# print(y)
# print(x[0, :])

"""3. 改变形状"""
# 同源改变，共享同一个内存
# y = x.view(15)
# z = x.view(-1, 5)   # -1这里就是不去确定可以通过另一个维度推出来
# print(x.size(), y.size(), z.size())
"""这里x如果发生改变，y，z都会发生改变
这并不难理解，view可以理解为仅仅改变了对这个张量的观察角度"""

# 这段代码对上述情况进行了说明
# x += 1
# print(x)
# print(y)

# 返还一个新的副本（不共享同一个内存）
# x_cp = x.clone().view(15)
# x -= 1
# print(x)
# print(x_cp)

# 按另一个常用函数item(),它可以将一个标量 Tensor 转化为 Python number:
# x = torch.randn(1)
# print(x)
# print(x.item())

"""4. 广播机制"""
# 上面我们已经对于两个形状相同的Tensor按元素进行计算，
# 那么对于两个形状不同的 Tensor 按元素运算，先适当复制使得Tensor相同后按元素运算
# x = torch.arange(1, 3).view(1, 2)
# print(x)
# y = torch.arange(1, 4).view(3, 1)
# print(y)
# print(x + y)




