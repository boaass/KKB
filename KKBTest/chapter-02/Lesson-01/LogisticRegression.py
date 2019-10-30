# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(theta, X):
    return 1 / (1 + np.exp(-inference(theta, X)))


# step 1: 构建预测模型
def inference(theta, X):
    return np.dot(theta, X.T)


# step 2: 构建损失函数
def eval_loss(theta, X, Y):
    A = sigmoid(theta, X)
    loss = -(np.sum(Y * np.log(A).T + (1 - Y) * np.log(1 - A).T)) / Y.size
    return loss


# step 3: 构建 计算梯度的方法
def gradient(X, Y, theta):
    return (sigmoid(theta, X) - Y) * X / Y.size


# step 4: 梯度下降
def cal_step_gradient(X, Y, theta, lr):
    theta -= lr * gradient(X, Y, theta)
    return theta


# 开始训练
def train(X, Y, lr, max_iter, openPlt=False):
    fig, ax = None, None
    if openPlt:
        plt.ion()
        fig, ax = plt.subplots()

    X = np.insert(X, 0, 1, 1)
    m, n = X.shape
    # theta = np.mat(np.zeros(n))
    theta = np.mat(np.array([-4, 0, 0], dtype='float64'))
    for i in range(max_iter):
        theta = cal_step_gradient(X, Y, theta, lr)

        temp_theta = np.array(theta).flatten()
        b, w = -temp_theta[:2] / temp_theta[2]
        print('w:{0}, b:{1}'.format(w, b))
        print('loss is {}'.format(eval_loss(theta, X, Y)))

        # 画图
        if openPlt:
            x = [X[:, 1].min(), X[:, 2].max()]
            f = lambda x: w * x + b
            y = [f(i) for i in x]

            positive = []
            negative = []
            temp_Y = np.array(Y).flatten()
            for j in range(len(temp_Y)):
                if temp_Y[j] == 1:
                    positive.append([X[j, 1], X[j, 2]])
                else:
                    negative.append([X[j, 1], X[j, 2]])

            positive = np.array(positive)
            negative = np.array(negative)

            ax.plot(positive[:, 0], positive[:, 1], 'rx')
            ax.plot(negative[:, 0], negative[:, 1], 'bo')

            ax.plot(x, y, c='red', label="regression line")
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc="lower right")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title(str(i) + ' iterations', fontsize='xx-large')
            plt.xlim(X[:, 1].min() - 5, X[:, 1].max() + 5)
            plt.ylim(X[:, 2].min() - 5, X[:, 2].max() + 5)
            plt.pause(0.01)
            ax.cla()

    return theta


# 预测
def predict(X, Y, theta):
    # 在矩阵 X 的第一列插入 1
    X = np.insert(X, 0, 1, 1)
    # 计算 X 样本对应的 Y = 1 的概率
    predictedY = np.array(sigmoid(theta, X)).flatten()
    gt_Y = np.array(Y).flatten()

    probit = 0
    for i in range(predictedY.size):
        if int(predictedY[i] + 0.5) == gt_Y[i]:
            probit += 1
    probit /= predictedY.size * 1.0
    print('Prediction success rate: {0}'.format(probit))

    return predictedY


def gen_sapmle_data():
    X = np.loadtxt("ex4x.dat")
    Y = np.loadtxt("ex4y.dat")
    return X, Y


# 绘制样本点
def plotsample(X, Y):
    positive = []
    negative = []
    for i in range(len(Y)):
        if Y[i] == 1: positive.append([X[i,0], X[i,1]])
        else: negative.append([X[i,0], X[i,1]])

    positive = np.array(positive)
    negative = np.array(negative)

    plt.plot(positive[:,0], positive[:,1], 'rx')
    plt.plot(negative[:,0], negative[:,1], 'bo')

if __name__ == '__main__':
    X, Y = gen_sapmle_data()
    matX = np.mat(X)
    matY = np.mat(Y)

    theta = train(matX, matY, 0.0014, 200000, True)
    # 测试
    predict(X, Y, theta)

    # 画图
    # plotsample(X, Y)
    # theta = np.array(theta).flatten()
    # b, a = -theta[:2] / theta[2]
    # f = lambda x: a * x + b
    # x = [X[:, 0].min(), X[:, 1].max()]
    # y = [f(i) for i in x]
    # plt.plot(x, y, c='red', label="regularized")
    # plt.legend(loc="lower right")
    # plt.xlim(X[:, 0].min()-0.1, X[:, 0].max() + 0.1)
    # plt.ylim(X[:, 1].min()-0.1, X[:, 1].max() + 0.1)
    # plt.xlabel("Feature 1")
    # plt.ylabel("Feature 2")
    # plt.title("Logistic Regression")
    # plt.show()
