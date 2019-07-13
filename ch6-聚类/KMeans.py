import numpy as np
import pandas as pd


class KMeans:
    # k: 聚类簇数
    def __init__(self, k=4):
        self.k = k

    # data: 待聚类数据
    # eps: 均值向量迭代收敛阈值
    def cluster(self, data, eps=1e-5):
        m, n = data.shape
        init = np.random.choice(range(m), self.k, replace=False)
        u = data[init, :]
        iter = 0
        while True:
            # 初始化k个类为空
            C = {}
            for i in range(self.k):
                C[str(i)] = []

            for j in range(m):
                min_d = np.inf
                index = -1
                for i in range(self.k):
                    d = np.linalg.norm(u[i] - data[j])
                    if d < min_d:
                        min_d = d
                        index = i
                C[str(index)].append(j)

            # 更新均值向量
            u_new = u.copy()
            for i in range(self.k):
                means = np.zeros(n)
                for t in C[str(i)]:
                    means = means + data[t]
                u_new[i] = means/len(C[str(i)])

            loss = np.linalg.norm(u_new - u, 'fro')

            print("iteration: ", iter)
            print("u old: ", u)
            print("u new: ", u_new)
            u = u_new

            print("Frobenius norm loss: ", loss)
            print("=" * 80)

            if loss <= eps:
                break
            iter += 1

        return C


def pre_processing():
    data = pd.read_csv('../datasets/iris.txt', delim_whitespace=True)
    data = np.array(data)
    x_data, y_data = data[:, :-1], data[:, -1]
    k = len(np.unique(y_data))
    return x_data, y_data, k


if __name__ == '__main__':
    data, label, k = pre_processing()
    cluster = KMeans(k)

    eps = 1e-2
    result = cluster.cluster(data)
    print("cluster result: ", result)
