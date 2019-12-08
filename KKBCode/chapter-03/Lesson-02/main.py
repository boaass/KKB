import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x[0] * x[0] + 50 * x[1] * x[1]


def g(x):
    return np.array([2 * x[0], 100 * x[1]])


def contour(X, Y, Z, arr=None):
    plt.figure(figsize=(10, 5))
    plt.contour(X, Y, Z, colors='black')
    plt.plot(0, 0, marker='*')

    if arr is not None:
        arr = np.array(arr)
        for i in range(len(arr) - 1):
            print(arr[i:i+2, 0])
            plt.plot(arr[i:i+2, 0], arr[i:i+2, 1])
    plt.show()


def gd(x_start, step, g):
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    for i in range(20):
        grad = g(x)
        x -= grad * step
        passing_dot.append(x.copy())
        if abs(sum(grad)) < 1e-6:
            break
    return x, passing_dot


def momentum(x_start, step, g, discount=0.7):
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    for i in range(20):
        grad = g(x)
        pre_grad = pre_grad * discount + grad
        x -= pre_grad * step
        passing_dot.append(x.copy())
        if abs(sum(pre_grad)) < 1e-6:
            break
    return x, passing_dot


def nesterov(x_start, step, g, discount=0.7):
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)

    for i in range(20):
        futurth_x = x - step * discount * pre_grad
        grad = g(futurth_x)
        pre_grad = pre_grad * discount + grad
        x -= pre_grad * step
        passing_dot.append(x.copy())
        if abs(sum(pre_grad)) < 1e-6:
            break

    return x, passing_dot


if __name__ == '__main__':
    xi = np.linspace(-200, 200, 1000)
    yi = np.linspace(-100, 100, 1000)
    X, Y = np.meshgrid(xi, yi)
    Z = X * X + 50 * Y * Y

    #
    # res, x_arr = gd([150, 75], 0.015, g)

    # momentum
    # res, x_arr = momentum([150, 75], 0.015, g)

    # nesterov
    res, x_arr = nesterov([150, 75], 0.013, g)

    contour(X, Y, Z, x_arr)
