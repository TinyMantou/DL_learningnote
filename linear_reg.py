# -*- coding: utf-8 -*-

from mxnet import nd, autograd
from IPython import display
from matplotlib import pyplot as plt
import random
import numpy

num_input = 2
num_example = 1000  # 数据样本数
true_w = [2, -3.4]  # 权重的标准
true_b = 4.2  # 偏移量标准
features = nd.random.normal(scale=1, shape=(num_example, num_input))  # 特征量：自变量
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # 标签：因变量
labels += nd.random.normal(scale=0.01, shape=labels.shape)  # 噪声
lr = 0.03  # 学习率,后面使用不同学习率检测效果
num_epochs = 3  # 迭代周期数
w = nd.random.normal(scale=0.01, shape=(num_input, 1))  # 标准差为scale
b = nd.zeros(shape=(1,))  # 偏移量为零
batch_size = 10
# 创建梯度
w.attach_grad()
b.attach_grad()

'''设置为矢量图模式'''


def use_svg_display():
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


'''从features和labels中获取batch_size个数据'''


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)


'''定义模型'''


def linreg(x, w, b):
    return nd.dot(x, w) + b


net = linreg

'''定义损失函数'''


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


loss = squared_loss

'''定义优化算法'''


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


'''参数训练'''


def train_wb(w, b):
    loss_list = []
    for epoch in range(num_epochs):
        #  优化参数 w,b
        for x, y in data_iter(batch_size, features, labels):  # y是个行向量
            with autograd.record():
                l = loss(net(x, w, b), y)  # 小批量x,y的损失
            l.backward()  # 小批量损失对参数模型求梯度
            sgd([w, b], lr, batch_size)  # 根据学习率来优化参数
        train_l = loss(net(features, w, b), labels)
        print('epoch: %d, loss: %f' % (epoch + 1, train_l.mean().asnumpy()))
        loss_list.append(train_l.mean().asnumpy())
    print('\n')
    return loss_list


def plot(x, y, condition):
    plt.plot(range(x), y, label='lr=' + str(condition))  # 加上标签
    plt.legend(loc='upper right')  # 绘制图例：左上方


for lr in numpy.arange(0.01, 0.05, 0.01):
    print('when learnrate = %f:' % lr)
    # 重置参数
    w = nd.random.normal(scale=0.01, shape=(num_input, 1))  # 标准差为scale
    b = nd.zeros(shape=(1,))  # 偏移量为零
    w.attach_grad()
    b.attach_grad()

    loss_list = train_wb(w, b)
    plot(num_epochs, loss_list, lr)

plt.show()
