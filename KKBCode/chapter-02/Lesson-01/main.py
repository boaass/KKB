# -*- coding:utf-8 -*-


import numpy as np
import random
import matplotlib.pyplot as plt
import time


def inference(w, b, x):
    # 模型, 预测 y
    pred_y = w * x + b
    return pred_y


def eval_loss(w, b, x_list, gt_y_list):
    # 损失函数
    # gt_y_list: 真实值
    avg_loss = 0
    for i in range(len(x_list)):
        avg_loss += 0.5 * (inference(w, b, x_list[i]) - gt_y_list[i]) ** 2
    avg_loss /= len(gt_y_list)
    return avg_loss


def gradient(pred_y, gt_y, x):
    # 计算梯度
    diff = pred_y - gt_y
    dw = diff * x
    db = diff
    return dw, db


def cal_step_gradient(batch_x_list, batch_y_list, w, b, lr):
    # 取部分数据算梯度
    avg_dw, avg_db = 0, 0
    batch_size = len(batch_x_list)
    for i in range(batch_size):
        pre_y = inference(w, b, batch_x_list[i])
        dw, db = gradient(pre_y, batch_y_list[i], batch_x_list[i])
        avg_dw += dw
        avg_db += db

    avg_dw /= batch_size
    avg_db /= batch_size
    w -= lr * avg_dw
    b -= lr * avg_db
    return w, b


def gen_sample_data():
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


def train(x_list, gt_y_list, batch_size, lr, max_iter):
    w = 0
    b = 0
    num_samples = len(x_list)
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(x_list), batch_size)  # 随机抽取 batch_size 个样本
        batch_x = [x_list[j] for j in batch_idxs]
        batch_y = [gt_y_list[j] for j in batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        print('w:{0}, b:{1}'.format(w, b))
        print('loss is {}'.format(eval_loss(w, b, x_list, gt_y_list)))
        time.sleep(1)

    return w, b


if __name__ == '__main__':
    x_list, y_list = gen_sample_data()
    # plt.scatter(x_list, y_list)
    # plt.show()

    train(x_list, y_list, 100, 0.001, 100)
