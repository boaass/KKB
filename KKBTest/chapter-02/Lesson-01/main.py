# -*- coding:utf-8 -*-
import time
import random
import numpy as np
import matplotlib.pyplot as plt

# Liner Regeression


# step 1: 构建预测模型
def inference(w, b, x):
    pred_y = w * x + b
    return pred_y


# step 2: 构建损失函数
def eval_loss(w, b, x_list, gt_y_list):
    loss = 0
    for i in range(len(x_list)):
        loss += (inference(w, b, x_list[i]) - gt_y_list[i]) ** 2

    loss /= 2 * len(x_list)
    return loss


# step 3: 构建 计算梯度的方法
def gradient(pred_y, gt_y, x):
    diff = pred_y - gt_y

    dw = diff * x
    db = diff
    return dw, db


# step 3: 梯度下降
def cal_step_gradient(batch_x_list, batch_y_list, w, b, lr):
    avg_dw, avg_db = 0, 0
    batch_size = len(batch_x_list)
    for i in range(batch_size):
        x = batch_x_list[i]
        gt_y = batch_y_list[i]
        pred_y = inference(w, b, x)
        dw, db = gradient(pred_y, gt_y, x)
        avg_dw += dw
        avg_db += db

    avg_dw /= batch_size
    avg_db /= batch_size

    w -= lr * avg_dw
    b -= lr * avg_db
    return w, b


# 生成测试数据
def gen_sapmle_data():
    w = random.randint(0, 10) + random.random()
    b = random.randint(0, 5) + random.random()

    num_sample = 100
    x_list = []
    y_list = []
    for i in range(num_sample):
        x = random.randint(0, 100) + random.random()
        y = inference(w, b, x) + random.random() * random.randint(-1, 100)

        x_list.append(x)
        y_list.append(y)

    return x_list, y_list


# 开始训练
def train(x_list, gt_y_list, batch_size, lr, max_iter):
    w = 0
    b = 0
    plt.ion()
    fig, ax = plt.subplots()
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(x_list), batch_size)
        batch_x = [x_list[j] for j in batch_idxs]
        batch_y = [gt_y_list[j] for j in batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        print('w:{0}, b:{1}'.format(w, b))
        print('loss is {}'.format(eval_loss(w, b, x_list, gt_y_list)))
        # 画图
        x_line = np.linspace(np.min(x_list), np.max(x_list), 1000)
        y_line = w * x_line + b
        ax.plot(x_line, y_line, c='r')
        plt.title(str(i) + ' iterations', fontsize='xx-large')
        ax.scatter(x_list, gt_y_list)
        plt.pause(1)
        ax.cla()

    return w, b


if __name__ == '__main__':
    x_list, y_list = gen_sapmle_data()
    train(x_list, y_list, 100, 0.0001, 100)