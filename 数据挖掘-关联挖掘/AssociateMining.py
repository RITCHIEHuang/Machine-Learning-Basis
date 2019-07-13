# %%
import numpy as np
import pandas as pd


class Apriori:
    def __init__(self, min_sup, data):
        self.min_sup = min_sup
        self.data = data
        self.answer = {}
        self.res = {}
        self.m_samples = len(data)

    def apriori_algorithm(self):
        k = 1
        f_initial = list()
        sup = np.sum(data, axis=0) / self.m_samples
        for i in range(len(sup)):
            if sup[i] >= self.min_sup:
                # print(i, sup[i])
                f_initial.append(i)

                self.answer[str(i)] = sup[i]

        self.res[str(k)] = f_initial
        while len(self.res[str(k)]) > 0:
            f_new = self.generate(self.res[str(k)])
            k += 1
            self.res[str(k)] = f_new
        # print(self.res)
        # print(self.answer)

    # 确保f_(k)元素按字典序存放
    # 由f_(k) 生成f_(k+1)
    def generate(self, f):
        res = list()
        for i in range(len(f) - 1):
            for j in range(i + 1, len(f)):
                sup, item = self.judge(f[i], f[j])
                if sup >= self.min_sup:
                    # print(item, sup)
                    res.append(item)
                    self.answer[self.to_str(item)] = sup
        return res

    def judge(self, a, b):
        if isinstance(a, int) and isinstance(b, int):
            return len(np.where((self.data[:, a] == 1) & (self.data[:, b] == 1))[0]) / self.m_samples, [a, b]
        else:
            assert len(a) == len(b)
            length = len(a)
            for k in range(length - 1):
                if a[k] != b[k]:
                    return -1, -1
            merge = a.copy()
            merge.append(b[length - 1])
            filter_data = np.take(data, merge, axis=1)
            cnt = 0
            for row in filter_data:
                flag = True
                for i in row:
                    if i != 1:
                        flag = False
                        break
                if flag:
                    cnt += 1
            return cnt / len(data), merge

    @staticmethod
    def to_str(key_list):
        r_str = ""
        for i in key_list:
            r_str += str(i) + " "
        return r_str[:-1]

    def print(self):
        sorted_res = sorted(self.answer.items(), key=lambda obj: obj[0])
        for item in sorted_res:
            print("%s %.3f" % (item[0], item[1]))


def preprocessing():
    raw_data = pd.read_csv('../datasets/data.txt', sep=' ', header=None)
    data = np.array(raw_data)[1:, :]
    return np.hstack((np.zeros((len(data), 1)), data))


if __name__ == "__main__":
    data = preprocessing()
    minsup = 0.144
    apriori = Apriori(minsup, data)
    apriori.apriori_algorithm()
    apriori.print()
