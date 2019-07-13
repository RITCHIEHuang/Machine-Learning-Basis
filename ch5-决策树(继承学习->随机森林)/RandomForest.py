# %%
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


# SVD++
# FM
# <<An Introduction to Computational Learning Theory>>

class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature = None
        self.feature_val = None
        self.label = None
        self.is_leaf = False

    @property
    def feature(self):
        return self._feature

    @feature.setter
    def feature(self, val):
        self._feature = val

    @property
    def feature_val(self):
        return self._feature_val

    @feature_val.setter
    def feature_val(self, val):
        self._feature_val = val

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def is_leaf(self):
        return self._is_leaf

    @is_leaf.setter
    def is_leaf(self, value):
        self._is_leaf = value

    @property
    def left_child(self):
        return self._left_child

    @left_child.setter
    def left_child(self, left):
        self._left_child = left

    @property
    def right_child(self):
        return self._right_child

    @right_child.setter
    def right_child(self, right):
        self._right_child = right

    @feature.setter
    def feature(self, value):
        self._feature = value

    def __str__(self):
        print(
            "label: {}, feature: {}, val: {}, lchild: {}, rchild: {}".format(self.label, self.feature, self.feature_val,
                                                                             str(self.left_child),
                                                                             str(self.right_child)))


# %%
# 使用bootstrap 在数据集D上进行随机采样k个数据，构成新的数据集
def bootstrap_sampling(d, k):
    return np.random.choice(d, k, replace=True)


def score(y, y_hat):
    return np.mean(y == y_hat)


class DecisionTree:
    def __init__(self, max_depth, split_limit, data):
        self.max_depth = max_depth
        self.split_limit = split_limit
        m, n = data.shape
        self.n_features = n - 1
        self.k = int(np.log2(n))
        self.tree = self.generate_tree(data, range(n - 1))

    # 实现决策树算法
    # 关注节点存储的信息：存储该节点的数据下标,数据的预测标签（叶子节点才有)
    # d: 数据集合
    # a: 属性集合（属性下标集合）
    def generate_tree(self, d, a, depth=1):
        # if depth > self.max_depth:
        #     return
        unique_y_label = np.unique(d[:, -1])
        node = Node()
        if len(unique_y_label) == 1:
            node.is_leaf = True
            node.label = unique_y_label[0]
            return node

        if a is None or len(a) == 0 or len(np.unique(np.take(d, a, axis=1))) == 1:
            node.is_leaf = True
            node.label = self.most_common_class(d)
            return node

        if len(d) <= self.split_limit:
            node.is_leaf = True
            node.label = self.most_common_class(d)
            return node

        # 随机森林每次划分时选择随机的a的子集
        a = bootstrap_sampling(self.n_features, self.k)

        # 从属性子集中取最佳属性划分
        best_attr, best_attr_val = self.select_best_feature(d, a)
        # print('best feature: ', best_attr, 'best feature value: ', best_attr_val)

        d_left = d[np.where(d[:, best_attr] <= best_attr_val)]
        d_right = d[np.where(d[:, best_attr] > best_attr_val)]
        # print('split', len(d_left), len(d_right))
        # print(best_attr)
        node.feature = best_attr
        node.feature_val = best_attr_val

        if d_left is None or len(d_left) == 0:
            node.left_child = Node()
            node.left_child.is_leaf = True
            node.left_child.label = self.most_common_class(d)
            return node
        else:
            node.left_child = self.generate_tree(d_left, a, depth + 1)

        if d_right is None or len(d_right) == 0:
            node.right_child = Node()
            node.right_child.is_leaf = True
            node.right_child.label = self.most_common_class(d)
            return node
        else:
            node.right_child = self.generate_tree(d_right, a, depth + 1)
        return node

    # 从属性集中选择最优属性
    # a: 属性集合下标
    def select_best_feature(self, d, a):
        best_gain = -np.inf
        best_attr = -1
        best_attr_val = -10
        for attr in a:
            cur_gain, cur_val = self.gain(d, attr)
            # print('cur gain: ', cur_gain, 'cur feature value: ', cur_val)
            if cur_gain > best_gain:
                best_attr = attr
                best_gain = cur_gain
                best_attr_val = cur_val

        return best_attr, best_attr_val

    def most_common_class(self, d):
        return Counter(d[:, -1]).most_common(1)[0][0]

    def calc_entropy(self, d):
        m = len(d)
        counter_dict = Counter(d[:, -1])
        ent = 0.0
        for v in counter_dict.values():
            ent -= (v / m) * np.log2(v / m)
        return ent

    # 最优化分算法: 信息增益
    # a: 属性
    def gain(self, d, a):
        old_entropy = self.calc_entropy(d)
        # x_data = np.take(self.X_data, d, axis=0)
        # 连续属性的处理
        sorted_attr_values = np.sort(d[:, a])
        middle_values = set()
        for i in range(len(sorted_attr_values) - 1):
            middle_values.add(1.0 * (sorted_attr_values[i] + sorted_attr_values[i + 1]) / 2)
        best_ent = np.inf
        best_val = 10.0
        for attr_val in middle_values:
            d_left = d[np.where(d[:, a] <= attr_val)]
            d_right = d[np.where(d[:, a] > attr_val)]
            cur_ent = 0.0
            cur_ent += len(d_left) / len(d) * self.calc_entropy(d_left) + len(d_right) / len(d) * self.calc_entropy(
                d_right)
            if cur_ent < best_ent:
                best_ent = cur_ent
                best_val = attr_val
        return old_entropy - best_ent, best_val

    def _predict(self, x, node):
        if node.label is not None:
            return node.label
        if x[node.feature] <= node.feature_val:
            return self._predict(x, node.left_child)
        if x[node.feature] > node.feature_val:
            return self._predict(x, node.right_child)

    def predict(self, X):
        y_hat = np.array([self._predict(x, self.tree) for x in X])
        return y_hat


class RandomForest:
    def __init__(self, data, num_trees=5, tree_max_depth=30, node_split_limit=10):
        self.n_trees = num_trees
        self.data = data
        self.tree_max_depth = tree_max_depth
        self.node_split_limit = node_split_limit
        self.trees = list()
        self.train()

    def train(self):
        for i in range(self.n_trees):
            m, n = data.shape
            bs = bootstrap_sampling(m, m)
            x_train = np.take(self.data, bs, axis=0)
            tree = DecisionTree(self.tree_max_depth, self.node_split_limit, x_train)
            print("Generate ", i + 1, " Decision Tree!")
            self.trees.append(tree)

    def predict(self, X_test, y_test):
        y_hats = np.array([tree.predict(X_test) for tree in self.trees])
        scores = [score(y_test, y_hat) for y_hat in y_hats]
        rf_accs = []
        for i in range(1, len(scores)+1):
            print('Tree ', i, ' Accuracy: ', scores[i-1])
            if i == 1:
                acc = scores[0]
            else:
                acc = np.mean(self.vote(y_hats[:i, :]) == y_test)
            print('number of Decision Tree: ', i, " Accuracy of Decision Tree: ", acc)
            rf_accs.append(acc)
        return rf_accs

    @staticmethod
    def vote(y_hat):
        # print(y_hat.shape)
        m, n = y_hat.shape
        # y_hat = y_hat.T
        y_hat = np.array([Counter(y_hat[:, col]).most_common(1)[0][0] for col in range(n)])
        # print(y_hat)
        return y_hat


def preprocessing():
    # %%
    raw_train_data = pd.read_csv('../datasets/mnistTrain_scale.txt', header=None)
    raw_test_data = pd.read_csv('../datasets/mnistTest_scale.txt', header=None)
    raw_train_data = np.array(raw_train_data)
    raw_test_data = np.array(raw_test_data)

    # %%
    raw_train_data = raw_train_data[np.where((raw_train_data[:, -1] == 1) | (raw_train_data[:, -1] == 7))]
    raw_test_data = raw_test_data[np.where((raw_test_data[:, -1] == 1) | (raw_test_data[:, -1] == 7))]
    # %%
    X_test, y_test = raw_test_data[:, :-1], raw_test_data[:, -1]
    return raw_train_data, X_test, y_test


def make_plot(x_data, y_data, title, x_label, y_label):
    plt.figure()
    plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    data, X_test, y_test = preprocessing()
    # for n in range(1, 10):
    n_trees = 10
    rf = RandomForest(data, n_trees, 30, 10)
    acc = rf.predict(X_test, y_test)
    make_plot(range(1, len(acc) + 1), acc, 'RandomForest Accuracy vs Number of Decision Tree',
              'number of Decision Tree', 'Accuracy')
