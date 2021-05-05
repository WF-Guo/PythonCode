"""
模拟退火算法：
测试函数：
f = 1 + (add_all(x[i]**2))/4000 - multiply_all(math.cos(x[i])/(math.sqrt(i)))

https://github.com/guofei9987/scikit-opt/

"""

import numpy as np
import time
import matplotlib.pyplot as plt

class SimulatedAnnealingBase():
    def __init__(self, func, x0, T_max=100, T_min=1e-7, L=300, max_stay_counter=150, **kwargs):
        assert T_max > T_min > 0, 'T_max > T_min > 0'

        self.func = func
        self.T_max = T_max  # initial temperature
        self.T_min = T_min  # end temperature
        self.L = int(L)  # num of iteration under every temperature（also called Long of Chain）
        # stop if best_y stay unchanged over max_stay_counter times (also called cooldown time)
        self.max_stay_counter = max_stay_counter

        self.n_dims = len(x0)

        self.best_x = np.array(x0)  # initial solution
        self.best_y = self.func(self.best_x)
        self.T = self.T_max
        self.iter_cycle = 0
        self.generation_best_X, self.generation_best_Y = [self.best_x], [self.best_y]
        # history reasons, will be deprecated
        self.best_x_history, self.best_y_history = self.generation_best_X, self.generation_best_Y

    def get_new_x(self, x):
        u = np.random.uniform(-1, 1, size=self.n_dims)
        x_new = x + 20 * np.sign(u) * self.T * ((1 + 1.0 / self.T) ** np.abs(u) - 1.0)
        return x_new

    def cool_down(self):
        self.T = self.T *0.7

    def isclose(self, a, b, rel_tol=1e-09, abs_tol=1e-30):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def run(self):
        x_current, y_current = self.best_x, self.best_y
        stay_counter = 0
        while True:
            for i in range(self.L):
                x_new = self.get_new_x(x_current)
                y_new = self.func(x_new)

                # Metropolis
                df = y_new - y_current
                if df < 0 or np.exp(-df / self.T) > np.random.rand():
                    x_current, y_current = x_new, y_new
                    if y_new < self.best_y:
                        self.best_x, self.best_y = x_new, y_new

            self.iter_cycle += 1
            self.cool_down()
            self.generation_best_Y.append(self.best_y)
            self.generation_best_X.append(self.best_x)

            # if best_y stay for max_stay_counter times, stop iteration
            if self.isclose(self.best_y_history[-1], self.best_y_history[-2]):
                stay_counter += 1
            else:
                stay_counter = 0

            if self.T < self.T_min:
                # stop_code = 'Cooled to final temperature'
                break
            if stay_counter > self.max_stay_counter:
                # stop_code = 'Stay unchanged in the last {stay_counter} iterations'.format(stay_counter=stay_counter)
                break

        return self.best_x, self.best_y,self.iter_cycle

#处理Griewank函数
def add_all(x):
    sum_all = 0
    for i in range(len(x)):
        sum_all += x[i]**2
    return sum_all


def multiply_all(x):
    multi_sum = 0
    for i in range(0,len(x)):
        multi_sum *= np.cos(x[i]) / np.sqrt(i+1)  #避免除以0
    return multi_sum


def func(x):
    return 1 + add_all(x)/4000 - multiply_all(x)


def SaveDate(it,z, time):
    data = list(zip(it,z))
    data.append('elapsed time:' + str(time))

    s = ''
    with open('SA_result.txt','w+') as w:
        for d in data:
            s += str(d) + '\n'
        w.writelines(s)


if __name__ == '__main__':
    x = np.random.uniform(-200,200,5)
    sa = SimulatedAnnealingBase(func, x)
    t1 = time.perf_counter()
    best_x, best_y, iter_cycle = sa.run()
    t2 = time.perf_counter()
    t = t2 - t1

    y = sa.generation_best_Y

    # it = []
    # for i in range(iter_cycle + 1):
    #     it.append(i)
    it = np.arange(0, sa.iter_cycle+1, 1)

    SaveDate(it, y, t)
    plt.plot(it, y)
    plt.savefig('SA_result.png')
    plt.show()
    print(f'best_x = {best_x}, best_y = {best_y},耗时:{t}')
