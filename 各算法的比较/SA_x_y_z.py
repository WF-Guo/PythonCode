"""
模拟退火算法：
测试函数在objectFunction里
"""

import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
class SimulatedAnnealingBase():
    """
    Parameters
    ----------------
    func : function
        The func you want to do optimal
    n_dim : int
        number of variables of func
    x0 : array, shape is n_dim
        initial solution
    T_max :float
        initial temperature
    T_min : float
        end temperature
    L : int
        num of iteration under every temperature（Long of Chain）
    ----------------------
    Examples ---> https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa.py
    e.g:
    demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
    sa = SA(func=demo_func, x0=[1, 1, 1], T_max=1, T_min=1e-9, L=300, max_stay_counter=150)
    best_x, best_y = sa.run()
    print('best_x:', best_x, 'best_y', best_y)

    """
    #原代码是计算一元函数的，我改成计算二元函数的

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
        self.best_y = np.array(x0)
        self.best_z = self.func(self.best_x,self.best_y)
        self.T = self.T_max
        self.iter_cycle = 0
        self.generation_best_X, self.generation_best_Y, self.generation_best_Z = [self.best_x], [self.best_y], [self.best_z]
        # history reasons, will be deprecated
        self.best_x_history, self.best_y_history, self.best_z_history = self.generation_best_X, self.generation_best_Y, self.generation_best_Z

    def get_new_x(self, x):
        u = np.random.uniform(-1, 1, size=self.n_dims)
        x_new = x + 20 * np.sign(u) * self.T * ((1 + 1.0 / self.T) ** np.abs(u) - 1.0)
        return x_new

    def get_new_y(self,y):
        u = np.random.uniform(-1, 1, size=self.n_dims)
        y_new = y + 20 * np.sign(u) * self.T * ((1 + 1.0 / self.T) ** np.abs(u) - 1.0)
        return y_new

    def cool_down(self):
        self.T = self.T_max/(1+self.T)

    def isclose(self, a, b, rel_tol=1e-09, abs_tol=1e-30):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def run(self):
        x_current, y_current, z_current = self.best_x, self.best_y, self.best_z
        stay_counter = 0
        while True:
            for i in range(self.L):
                x_new = self.get_new_x(x_current)
                y_new = self.get_new_y(y_current)
                z_new = self.func(x_new,y_new)

                # Metropolis
                df = z_new - z_current
                if df < 0.0 :
                    x_current, y_current, z_current = x_new, y_new , z_new
                else:
                    if np.exp(-df / self.T) > np.random.rand():
                        x_current, y_current, z_current = x_new, y_new, z_new
                if z_new < self.best_z:
                    self.best_x, self.best_y,self.best_z = x_new, y_new, z_new

            self.iter_cycle += 1
            self.cool_down()
            self.generation_best_Y.append(self.best_y)
            self.generation_best_X.append(self.best_x)
            self.generation_best_Z.append(self.best_z)


            # if best_y stay for max_stay_counter times, stop iteration
            if self.isclose(self.best_z_history[-1], self.best_z_history[-2]):
                stay_counter += 1
            else:
                stay_counter = 0

            if self.T < self.T_min:
                # print('Cooled to final temperature')
                break
            if stay_counter > self.max_stay_counter:
                # print('Stay unchanged in the last %d iterations'%stay_counter)
                break

        return self.best_x, self.best_y, self.best_z , self.iter_cycle


#  rastrigin测试函数
def func(X, Y):
    A = 10
    for i in range(len(X)):
        Z1 = 2 * A + X[i] ** 2 - A * np.cos(2 * np.pi * X[i])
    for i in range(len(Y)):
        Z2 = Y[i] ** 2 - A * np.cos(2 * np.pi * Y[i])
    Z = Z1 +Z2
    return Z

def SaveDate(it,z, time):
    data = list(zip(it,z))
    data.append('elapsed time:' + str(time))

    s = ''
    with open('SA_result_x_y_z.txt','w+') as w:
        for d in data:
            s += str(d) + '\n'
        w.writelines(s)

if __name__ == '__main__':
    x0 = np.arange(-10,10,0.1)
    sa = SimulatedAnnealingBase(func,x0)
    t1 = time.perf_counter()
    best_x, best_y, best_z, iter_cycle = sa.run()
    t2 = time.perf_counter()
    t = t2-t1

    z = sa.generation_best_Z

    it = []
    for i in range(iter_cycle+1):
        it.append(i)

    SaveDate(it,z,t)
    plt.plot(it,z)
    plt.savefig('SA_result_x_y_z.png')
    plt.show()
    # print(len(z),iter_cycle)

    print(f'best_x = {best_x}, best_y = {best_y}, best_z = {best_z},耗时:{t}')