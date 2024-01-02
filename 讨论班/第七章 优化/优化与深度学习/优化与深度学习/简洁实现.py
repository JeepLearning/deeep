# 简洁实现

import numpy as np
import time
import torch
from torch import nn, optim
from 小批量随机梯度下降 import get_data_ch7
import sys
import d2lzh_pytorch as d2l
import torch.utils.data

features, labels = get_data_ch7()


def train_pytorch_ch7(optimizer_fn, optimizer_hyperparams, features, labels,
                      batch_size=10, num_epochs=2):
    # 初始化模型
    # nn.sequential--->--->封装
    net = nn.Sequential(nn.Linear(features.shape[-1], 1))  # shape 读取矩阵维度
    # MSELoss
    loss = nn.MSELoss()
    # print(net.parameters())  # 直接整这个返回的是地址
    # # 输出网络参数
    # paras = list(net.parameters())
    # for w, b in enumerate(paras):
    #     print('w: ', num)
    #     print('b: ', para)
    #     print('----------------------------------')

    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)  # **表示分配字典，算法参数

    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels),
        batch_size, shuffle=True)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            # 除以2是为了和train_ch7保持⼀致, 因为squared_loss中除了2
            l = loss(net(X).view(-1), y) / 2

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')
    d2l.plt.show()


# 直接利用pytorch提供的优化器方法来训练模型
print('小批量随机梯度下降算法')
train_pytorch_ch7(optim.SGD, {'lr': 0.05}, features, labels, 10)  # {"lr": 0.05}
# train_ch7(optim.SGD, {'lr': 0.05}, features, labels, 10)  # {"lr": 0.05}
print('动量法')
train_pytorch_ch7(torch.optim.SGD, {'lr': 0.05, 'momentum': 0.9}, features, labels)
print('AdaGrad')
train_pytorch_ch7(torch.optim.Adagrad, {'lr': 0.1}, features, labels)
print('RMSprop算法')
train_pytorch_ch7(torch.optim.RMSprop, {'lr': 0.01, 'alpha': 0.9}, features, labels)
print('AdaDelta算法')
train_pytorch_ch7(torch.optim.Adadelta, {'rho': 0.9}, features, labels)
print('Adam算法')
train_pytorch_ch7(torch.optim.Adam,  {'lr': 0.01}, features, labels)

