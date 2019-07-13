import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

'''
BootStrap 方法的实现:
实现样本的有放回的取样，样本数量为N, 则算法做N次抽样。
抽样中选取的不重复的样本作为训练集，剩余的样本作为测试集。
这种策略不适合大样本的情况，而且难以保证样本的层次性。
'''


class BootStrap:
    def __init__(self):
        pass

    def generate_index(self, num):
        slice = []
        while (len(slice) < num):
            p = random.randrange(0, num)
            slice.append(p)
        return slice

    def train_test_split(self, X, y):
        assert len(X) == len(y)

        samples_num = len(X)

        all_index = set(range(samples_num))
        train_index = self.generate_index(samples_num)
        test_index = list(all_index - set(train_index))

        X_train = []
        X_test = []
        y_train = []
        y_test = []

        for i in train_index:
            X_train.append(X[i])
            y_train.append(y[i])

        for j in test_index:
            X_test.append(X[j])
            y_test.append(y[j])

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # load datasets from file
    print('starting loading data ...')
    data = pd.read_csv('../../datasets/iris.data', header=None)
    y = data.iloc[:, 4].values
    X = data.iloc[:, [2, 3]].values

    le = LabelEncoder()
    y = le.fit_transform(y)
    print('loading data finished ...')

    # Perform BootStrap Evaluation Method and LogisticRegression
    print('starting perform bootstrap method ... ')
    iter_n = 10
    scores = []

    lr = LogisticRegression(solver='lbfgs', multi_class='auto', C=100.0, random_state=1)
    bs = BootStrap()

    for i in range(iter_n):
        X_train, X_test, y_train, y_test = bs.train_test_split(X, y)

        lr.fit(X_train, y_train)
        score = lr.score(X_test, y_test)
        scores.append(score)
        print('iter {} - Accuracy: {}'.format(i + 1, score))

    print('BootStrap: Accuracy: {} +/- {}'.format(np.mean(scores), np.std(scores)))
