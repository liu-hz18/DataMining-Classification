import torch, random
import numpy as np

class mySVMModel:
    def __init__(self, epoch=10, kernel='linear', random_state=42):
        self.epoch = epoch
        self._kernel = kernel
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        np.random.seed(random_state)
        random.seed(random_state)

    # 核函数
    def kernel(self, x1, x2):
        if self._kernel == 'linear':
            return sum([x1[k] * x2[k] for k in range(self.n)])
        elif self._kernel == 'poly':
            return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1) ** 2
        return 0

    # g(x)预测值，输入xi（X[i]），g(x) = \sum_{j=1}^N {\alpha_j*y_j*K(x_j,x)+b}
    def _g(self, i):
        r = self.b
        for j in range(self.m):
            r += self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])
        return r

    # E（x）为g(x)对输入x的预测值和y的差，E_i = g(x_i) - y_i
    def _E(self, i):
        return self._g(i) - self.Y[i]

    def KKT(self, i):
        y_g = self._g(i) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1

    def init_alpha(self):
        # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)
        # 外层循环选择满足0<alpha_i<C，且不满足KKT的样本点。如果不存在遍历剩下训练集
        for i in index_list:
            if self.KKT(i):
                continue
            # 内层循环，|E1-E2|最大化
            E1 = self.E[i]
            # 如果E1是+，选择最小的E_i作为E2；如果E1是负的，选择最大的E_i作为E2
       
            if E1 >= 0:
                j = min(range(self.m), key=lambda x: self.E[x])
            else:
                j = max(range(self.m), key=lambda x: self.E[x])
            return i, j

    def _compare(self, _alpha, L, H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    def init_E(self):
        tot = np.zeros(self.n)
        for i in range(self.m):
            for j in range(self.n):
                tot[j] += self.Y[i] * self.X[i][j]
        E = np.zeros(self.m)
        for i in range(self.m):
            for j in range(self.n):
                E[i] += self.X[i][j] * tot[j]
            E[i] -= self.Y[i]
        return E

    def fit(self, features, labels):
        features, labels = np.array(features), np.array(labels)
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0

        # 将Ei保存在一个列表里
        self.alpha = np.ones(self.m)
        # self.E = [self._E(i) for i in range(self.m)]
        self.E = self.init_E()
        # 松弛变量
        self.C = 1.0

        for e in range(self.epoch):
            i1, i2 = self.init_alpha()

            # 边界,计算阈值b和差值E_i
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]
            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(self.X[i2], self.X[i2]) - 2 * self.kernel(self.X[i1], self.X[i2])
            if eta <= 0: continue
            # 更新约束方向的解
            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E1 - E2) / eta  #此处有修改，根据书上应该是E1 - E2，书上130-131页
            alpha2_new = self._compare(alpha2_new_unc, L, H)

            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2] - alpha2_new)

            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (alpha1_new - self.alpha[i1]) - self.Y[i2] * self.kernel(self.X[i2], self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) - self.Y[i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2

            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)

    def predict(self, datas):
        res = []
        tot = np.zeros(self.n)
        for i in range(self.m):
            for j in range(self.n):
                tot[j] += self.alpha[i] * self.Y[i] * self.X[i][j]
        for data in datas:
            r = self.b
            # for i in range(self.m):
            #     r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
            for j in range(self.n):
                r += tot[j] * data[j]
            res.append(1 if r > 0 else 0)
        return np.array(res)
