import torch
from torch import nn  # nn神经网络相关
from torch.nn import functional as F  # funticonal神经网络相关常用的函数
from torch import optim  # 优化

import torchvision  # 视觉工具包
from matplotlib import pyplot as plt

from utils import plot_image, plot_curve, one_hot

"""
mnist，手写数字识别分为四个步骤：
1、加载数据
2、建立模型
3、训练
4、测试
"""

batch_size = 512

"""
step1. load dataset

torchvision.datasets.MNIST：下载MNIST数据集到mnist_data下
train：表示是否为训练数据
download：表示是否下载了，未下载则自动下载
torchvision.transforms.ToTensor()：将numpy格式转为tensor
torchvision.transforms.Normalize()：正则化，神经网络接收的数据最好为以某个值均匀的分布，便于神经网络优化，不使用也可以但是会影响最终性能
batch_size：每次加载多少个数据
shuffle：数据随机打散
"""
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image sample')

"""
构建网络
"""
class Net(nn.Module):

    # 初始化网络
    def __init__(self):
        super(Net, self).__init__()

        # xw+b
        # 构建不同的层
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x: [b, 1, 28, 28]
        # h1 = relu(xw1+b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3+b3
        x = self.fc3(x)

        return x


net = Net()
# [w1, b1, w2, b2, w3, b3]
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train_loss = []

# 完成3次迭代
for epoch in range(3):

    # 完成一次就是对所有数据完成一次迭代
    for batch_idx, (x, y) in enumerate(train_loader):

        # x: [b, 1, 28, 28], y: [512]
        # [b, 1, 28, 28] => [b, 784]
        # x.size(0)就是batch size
        x = x.view(x.size(0), 28 * 28)
        # => [b, 10]
        out = net(x)
        # [b, 10]
        y_onehot = one_hot(y)
        # loss = mse(out, y_onehot)
        loss = F.mse_loss(out, y_onehot)

        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 梯度更新
        # w' = w - lr*grad
        optimizer.step()

        # 保存记录loss，此处loss是tensor数据类型
        train_loss.append(loss.item())

        # 每隔10个数据打印相关数据
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

# 画出train_loss图像
plot_curve(train_loss)
# we get optimal [w1, b1, w2, b2, w3, b3]

# 衡量准确度
total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28 * 28)
    out = net(x)
    # out: [b, 10] => pred: [b]
    # 1维度上的最大值为预测值
    pred = out.argmax(dim=1)
    # 记录一次迭代中所有预测正确的个数
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)

x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28 * 28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')
