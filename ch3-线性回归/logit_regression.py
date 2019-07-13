import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def loss_function(X, y, beta):
    num_data = X.shape[0]
    loss = 0.0
    for i in range(num_data):
        loss += -y[i] * np.dot(beta.T, X[i, :]) + np.log(1.0 + np.exp(np.dot(beta.T, X[i, :])))
    return loss


def sigmoid_function(z):
    return 1.0 / (1 + np.exp(-z))


def make_plot(x_data, y_data, title, x_label, y_label):
    plt.plot(x_data, y_data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# 训练函数，根据梯度下降迭代参数beta
def train(X, y, alpha, iterations):
    row, col = X.shape
    Beta = np.zeros(col)
    iteration = 0
    train_loss = []
    while iteration < iterations:
        gradient_beta = np.zeros(col)
        for i in range(row):
            loss = y[i] - sigmoid_function(np.dot(Beta.T, X[i, :]))
            gradient_beta -= X[i, :] * loss
        Beta -= alpha * gradient_beta
        likelihood_tmp = loss_function(X_train, y_train, Beta)
        train_loss.append(likelihood_tmp)
        print('iter: ', iteration, 'loss: ', likelihood_tmp)
        iteration += 1

    # 绘制迭代过程中的loss曲线
    make_plot(range(len(train_loss)), train_loss, "Loss Over Iterations", "iteration num", "loss")
    return Beta


def accuracy(X, y, beta):
    row, _ = X.shape
    y_predict = np.dot(X, beta)
    y_predict[sigmoid_function(y_predict) <= 0.5] = 0
    y_predict[sigmoid_function(y_predict) > 0.5] = 1

    cnt_correct = np.sum(y_predict == y)
    return 1.0 * cnt_correct / row


def preprocessing():
    train = pd.read_csv('../datasets/mnistTrain_scale.txt', header=None)
    test = pd.read_csv('../datasets/mnistTest_scale.txt', header=None)

    # 统计各标签出现的次数
    train = np.array(train)
    test = np.array(test)

    num_labels = np.zeros(10)
    for i in range(10):
        for row in train:
            if row[-1] == i:
                num_labels[i] += 1
    print(num_labels)

    rank_labels = np.argsort(-num_labels)
    print(rank_labels)

    # 找到标签为1，7的数据
    train_data = train[np.where((train[:, -1] == 1) | (train[:, -1] == 7))]
    test_data = test[np.where((test[:, -1] == 1) | (test[:, -1] == 7))]

    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    y_train[y_train == 1] = 0
    y_train[y_train == 7] = 1
    y_test[y_test == 1] = 0
    y_test[y_test == 7] = 1

    X_train = np.insert(X_train, 0, 1, axis=1)
    X_test = np.insert(X_test, 0, 1, axis=1)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = preprocessing()
    # 使用训练得到的beta对测试集进行预测评分
    # 设定训练参数alpha和迭代轮数max_iter
    alpha = 0.005
    max_iter = 500
    beta = train(X_train, y_train, alpha, max_iter)

    acc = accuracy(X_test, y_test, beta)
    print("Accuracy: ", acc)
