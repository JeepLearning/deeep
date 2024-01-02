import torch

"""设置如何追踪一个张量"""
# 这里我们定义一个张量, requires_grad 指定是否需要追踪在该<张量>的所有操作, 这对计算梯度是很有必要的[直接打印变脸可以看出来]
# x = torch.ones((2, 2), requires_grad=True)
# print(x)
# 这里查看的是张量是不是通过其他张量通过计算得来, 如果是则返还一个与运算相关的对象[后续会有提到], 否则返还None
# print(x.grad_fn)

# y = x + 2
# print(y)
# print(y.grad_fn)    # 这里y是有grad_fn的, 因为他是根据 x 创建的[AddBackward 代表通过加法得来]

# 根据是否直接创建, 将直接创建产生的称为<叶子节点>
# print(x.is_leaf)
# print(y.is_leaf)

# 考虑一些复杂运算
# z = y * y * 3
# out = z.mean()
# print(z, out)   # 可以看出, 在输出张量的同时对 grad_fn 也有记录

# 加入我们要后续指定是否需要追踪可以采用 .requires_grad_(True) 方法进行改变
# a = torch.randn((2, 2))     # 默认 requires_grad = False
# a = ((a * 3) / (a - 1))
# print(a.requires_grad)  # False
# a.requires_grad_(True)
# print(a.requires_grad)  # True
# b = (a * a).sum()
# print(b.grad_fn)

"""在确定追踪后, 计算梯度"""
# x = torch.ones((2, 2), requires_grad=True)
# y = x + 2
# z = y * y * 3
# out = z.mean()
# print(out)
# 由于计算出来的 out 是个标量, 所以不需要指定参数, 具体运算过程可以查看 P39
# out.backward()
# out.backward(torch.tensor(1.))    # 和上述等价
# print(x.grad)

# <反向传播是累加的, 每次反向传播都要累加>, 因此一般需要清空
# out2 = x.sum()
# out2.backward()
# print(x.grad)    # 这里加上了out反向传播时的梯度

# out3 = x.sum()
# x.grad.data.zero_()
# out3.backward()
# print(x.grad)   # 这里清零以后就只计算了 out3 中 x 的操作

"""
我们不允许张量对张量求导, 只允许标量对张量求导, 求导结果时同自变量同形的张量
y.backward(w)的含义是:先计算 l = torch.sum(y * w), 则 l 是个标量,  然后求 l 对 x 的导数
w 只是一个系数, 在 y 对 x 在 x = [x1, x2, ...,xn]处的求导数, 分别乘以 w.
而返还的结果于 x 的形状相同
"""
# 看一些实际例子
# x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
# y = 2 * x
# z = y.view((2, 2))
# print(z)

# 此时 y 不是一个标量, 在调用backward的时候需要传入一个形如 y 的权重向量加权得到一个标量
# v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
# z.backward(v)
# print(x.grad)

# 下面我们列举一盒梯度中断的例子
# x = torch.tensor(1.0, requires_grad=True)
# y1 = x ** 2
# with torch.no_grad():
#     y2 = x ** 3
# y3 = y1 + y2

# print(x.requires_grad)
# print(y1, y1.requires_grad)     # True
# print(y2, y2.requires_grad)     # False
# print(y3, y3.requires_grad)     # True

# 我们可以看出 y2 中没有 grad_fn 并且 y2.requires_grad = False, 而 y3 则全部拥有, 那么对 y3 求 x 的梯度会怎么样
# y3.backward()
# print(x.grad)   # 这是因为 y1 有关的梯度并不会回传, 只与 y1 的梯度才会回传
# 因此, y2.backward() 会报错

# 如果我们修改 tensor 的值但不想被记录, 那么我们可以对 tensor.data 进行操作
# x = torch.ones(1, requires_grad=True)

# print(x.data)   # 这里还是一个 tensor
# print(x.data.requires_grad)     # 独立出去了, 这里时False

# y = 2 * x
# x.data *= 100   # 这改变了值, 不会影响计算梯度

# y.backward()
# print(x)    # 但是更改也只会影响 tensor 的值
# print(x.grad)





