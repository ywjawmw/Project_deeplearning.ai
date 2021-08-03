# -*- coding: utf-8 -*-
# @Time    : 2021/7/7 10:51
# @Author  : Ywj
# @File    : 7.7hw_GD.py
# @Description : 7.7 homework  Gradient

"""
设置基本学习率为0.01，手写实现以下几种优化器：
1、SGD
2、Momentum
3、RMSprop
4、Adam
优化方法中所涉及的其他超参数自定，

初始化 (W, b) = (0, 0)
以SGD方法迭代100次所达到的损失值为基准，比较其他几种优化方法达到该值所需要的迭代轮次。
"""
import numpy as np

X = [0.0, 0.5, 0.8, 1.1, 1.5, 1.9, 2.2, 2.4, 2.6, 3.0]
Y = [0.9, 2.1, 2.7, 3.1, 4.1, 4.8, 5.1, 5.9, 6.0, 7.0]
X = np.array(X)
Y = np.array(Y)
X = X.reshape((X.shape[0], 1))
Y = Y.reshape((Y.shape[0], 1))
print("X.shape:", X.shape)
print("Y.shape:", Y.shape)


def cost_function(y_hat, y):
    """
    compute cost function
    :param y_hat:
    :param y:
    :return:
    """
    n = y_hat.shape[0]
    loss = np.sum(np.square(y - y_hat)) / 2 / n
    return loss

def forward(x, W, b):
    y_hat = W * x + b
    return y_hat

def init_parameter(n_0, n_1):
    W = np.zeros((n_0, n_1))
    b = np.zeros((n_0, 1))
    return W, b

def backward(x, y, W, b):
    n = x.shape[0]
    y_hat = forward(x, W, b)
    loss = cost_function(y_hat, y)
    df = y_hat - y
    dW = np.dot(df.T, x) / n
    db = np.sum(df, axis=0, keepdims=True) / n
    # print("n:", n)
    # print("dW.shape:", dW.shape)
    # print("db.shape:", db.shape)
    return dW, db, loss


def SGD(X, Y, chose_sgd, alpha=0.01, max_epoch=100):
    """
    :param X:
    :param Y:
    :param alpha:
    :param max_epoch:
    :return:
    """
    W, b = init_parameter(1, 1)
    n = X.shape[0]
    for epoch in range(max_epoch):
        if chose_sgd is True:
            # 使用随机梯度下降，随机取1个样本
            choose_i = np.random.randint(0, n)
            dW, db, cost = backward(X[choose_i: choose_i+1], Y[choose_i: choose_i+1], W, b)
        else:
            dW, db, cost = backward(X, Y, W, b)
        W -= alpha * dW
        b -= alpha * db
        # print("epoch: %d, cost is %.5f" % (epoch, cost))
    print("SGD cost is %.5f" % cost)
    return cost


def Momentum(X, Y, cost_sgd, alpha=0.01, beta=0.9):
    epoch = 0
    W, b = init_parameter(1, 1)
    dW, db, cost = backward(X, Y, W, b)
    v_dW = np.zeros((dW.shape[0], dW.shape[1]))
    v_db = np.zeros((db.shape[0], db.shape[1]))
    while cost > cost_sgd:
        epoch += 1
        v_dW = beta * v_dW + (1-beta) * dW
        v_db = beta * v_db + (1 - beta) * db
        W -= alpha * v_dW
        b -= alpha * v_db
        dW, db, cost = backward(X, Y, W, b)
        # print("epoch: %d, cost is %.5f" % (epoch, cost))
    print("Momentum 迭代%d次， 达到loss：%.5f" % (epoch, cost))


def RMSprop(X, Y, cost_sgd, alpha=0.01, beta=0.999):
    epoch = 0
    W, b = init_parameter(1, 1)
    dW, db, cost = backward(X, Y, W, b)
    S_dW = np.zeros((dW.shape[0], dW.shape[1]))
    S_db = np.zeros((db.shape[0], db.shape[1]))
    while cost > cost_sgd:
        epoch += 1
        S_dW = beta * S_dW + (1 - beta) * np.square(dW)
        S_db = beta * S_db + (1 - beta) * np.square(db)
        W -= alpha * dW / np.sqrt(S_dW)
        b -= alpha * db / np.sqrt(S_db)
        dW, db, cost = backward(X, Y, W, b)
        # print("epoch: %d, cost is %.5f" % (epoch, cost))
    print("RMSprop 迭代%d次， 达到loss：%.5f" % (epoch, cost))


def Adam(X, Y, cost_sgd, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    epoch = 0
    W, b = init_parameter(1, 1)
    dW, db, cost = backward(X, Y, W, b)
    v_dW = np.zeros((dW.shape[0], dW.shape[1]))
    v_db = np.zeros((db.shape[0], db.shape[1]))
    S_dW = np.zeros((dW.shape[0], dW.shape[1]))
    S_db = np.zeros((db.shape[0], db.shape[1]))
    while cost > cost_sgd:
        epoch += 1
        v_dW = beta1 * v_dW + (1 - beta1) * dW
        v_db = beta1 * v_db + (1 - beta1) * db
        S_dW = beta2 * S_dW + (1 - beta2) * np.square(dW)
        S_db = beta2 * S_db + (1 - beta2) * np.square(db)
        vdW_corrected = v_dW / (1 - np.power(beta1, epoch))
        vdb_corrected = v_db / (1 - np.power(beta1, epoch))
        SdW_corrected = S_dW / (1 - np.power(beta2, epoch))
        Sdb_corrected = S_db / (1 - np.power(beta2, epoch))
        W -= alpha * vdW_corrected / (np.sqrt(SdW_corrected) + epsilon)
        b -= alpha * vdb_corrected / (np.sqrt(Sdb_corrected) + epsilon)
        dW, db, cost = backward(X, Y, W, b)
        # print("epoch: %d, cost is %.5f" % (epoch, cost))
    print("Adam 迭代%d次， 达到loss：%.5f" % (epoch, cost))

sgd_cost = SGD(X, Y, False)
Momentum(X, Y, sgd_cost)
RMSprop(X, Y, sgd_cost)
Adam(X, Y, sgd_cost)