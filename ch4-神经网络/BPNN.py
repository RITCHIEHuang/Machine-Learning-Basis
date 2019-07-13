import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid_function(x):
    return 1.0 / (1 + np.exp(-x))


def loss(y, y_hat):
    return np.mean((y - y_hat) ** 2)


def loss_mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))


# 生成元素在-0.1到0.1之间的随机矩阵
# w为矩阵的维度，类型为tuple
def make_matrix(w):
    return -0.1 + 0.2 * np.random.random(w)


def make_plot(x_data, y_data, title, x_label, y_label):
    plt.figure()
    plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.legend()
    plt.show()


# t: 神经元个数
# X: (m, n)
# v: (n, t)
# hidden_input: (m, t)
# hidden_activation: (m, t)
# w: (t, 1)
# y_hat: (m, 1)
# gradient_w: (t, 1)
def train(X, y, t, alpha, max_iter):
    m, n = X.shape
    v = make_matrix((n, t))
    w = make_matrix((t, 1))

    loss_history = []
    for iteration in range(max_iter):
        hidden_input = np.dot(X, v)
        hidden_activation = sigmoid_function(hidden_input)

        y_hat = np.dot(hidden_activation, w)
        iter_loss = loss(y, y_hat)
        loss_history.append(iter_loss)
        print('iter: ', iteration, ' MSE: ', iter_loss)

        gradient_w = 1.0 / m * np.dot(hidden_activation.T, y - y_hat)
        w += alpha * gradient_w

        gradient_v = 1.0 / m * np.dot(X.T, hidden_activation * (1 - hidden_activation) * np.dot(y - y_hat, w.T))
        v += alpha * gradient_v

    make_plot(np.arange(len(loss_history)), loss_history, 'MSE during iteration', 'iteration', 'MSE')
    return v, w


def predict(X, y, v, w):
    hidden_input = np.dot(X, v)
    hidden_activation = sigmoid_function(hidden_input)

    y_hat = np.dot(hidden_activation, w)
    return y_hat


if __name__ == '__main__':
    data = pd.read_csv('../datasets/housing-data-new.txt', delim_whitespace=True, header=None)
    data = np.array(data)

    X_train, y_train = data[:406, :-1], data[:406, -1]
    X_test, y_test = data[406:, :-1], data[406:, -1]
    y_train = y_train.reshape((-1, len(y_train))).T
    y_test = y_test.reshape((-1, len(y_test))).T
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    m, n = X_train.shape
    for i in range(n):
        max_val, min_val = np.max(X_train[:, i]), np.min(X_train[:, i])
        print(max_val, min_val)
        X_train[:, i] = (X_train[:, i] - min_val) / (max_val - min_val)
        X_test[:, i] = (X_test[:, i] - min_val) / (max_val - min_val)

    X_train = np.hstack((X_train, np.ones((m, 1))))

    r, c = X_test.shape
    X_test = np.hstack((X_test, np.ones((r, 1))))

    # 设定训练参数
    alpha = 0.05
    hidden_size = np.arange(4, 40)
    iteration = 10000
    h_size = 7
    # loss_on_hidden_size = []
    # for h_size in hidden_size:
    v, w = train(X_train, y_train, h_size, alpha, iteration)

    # 测试集上预测
    y_test_hat = predict(X_test, y_test, v, w)
    mae = loss_mae(y_test, y_test_hat)
    # loss_on_hidden_size.append(mae)
    print('MAE on test sets: ', mae)

    # make_plot(hidden_size, loss_on_hidden_size, 'MAE over Neurons number', 'Neurons Num', 'MAE')
