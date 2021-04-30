"""
PSO算法：
测试函数：
Griewank函数
f = 1 + (add_all(x[i]**2))/4000 - multiply_all(math.cos(x[i])/(math.sqrt(i)))

"""
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
import csv


#处理Griewank函数
def add_all(x):
    sum_all = 0
    for i in range(len(x)):
        sum_all += x[i]**2
    return sum_all


def multiply_all(x):
    multi_sum = 0
    for i in range(0,len(x)):
        multi_sum *= math.cos(x[i]) / math.sqrt(i+0.0001)  #避免除以0
    return multi_sum


def object_function(x):
    return 1 + add_all(x)/4000 - multiply_all(x)



def controlRange(x):
    global value_up_range, value_down_range
    if x > value_up_range:
        while x > value_up_range:
            x -= value_up_range - value_down_range
    elif x <= value_down_range:
        while x < value_down_range:
            x += value_up_range - value_down_range
    return x


def controlAllXRange(xx):
    for i in range(0, len(xx)):
        for j in range(0, len(xx[i])):
            xx[i][j] = controlRange(xx[i][j])
    return xx


class PSO(object):
    def __init__(self, dim, population_size, max_steps):
        self.w = 0.6  # 惯性权重
        self.c1 = self.c2 = 2.0
        self.population_size = population_size  # 粒子群数量
        self.dim = dim  # 搜索空间的维度
        self.max_steps = max_steps  # 迭代次数
        self.x_bound = [-200, 200]  # 解空间范围
        self.x = np.random.uniform(self.x_bound[0], self.x_bound[1],
                                   (self.population_size, self.dim))  # 初始化粒子群位置
        self.v = np.random.rand(self.population_size, self.dim)  # 初始化粒子群速度
        fitness = self.calculate_fitness(self.x)
        self.p = self.x  # 个体的最佳位置
        self.pg = self.x[np.argmin(fitness)]  # 全局最佳位置
        self.individual_best_fitness = fitness  # 个体的最优适应度
        self.global_best_fitness = np.min(fitness)  # 全局最佳适应度
        self.all_fitness = []

    def calculate_fitness(self, data):
        f_list = []
        for x in data:
            # j 是 一列数据
            f = object_function(x)
            f_list.append(f)
        return np.array(f_list)

    def evolve(self):
        fig = plt.figure()
        for step in range(self.max_steps):

            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            # 更新速度和权重

            self.v = self.w * self.v + self.c1 * r1 * (self.p - self.x) + self.c2 * r2 * (self.pg - self.x)
            self.v = controlAllXRange(self.v)
            self.x = self.v + self.x
            self.x = controlAllXRange(self.x)

            fitness = self.calculate_fitness(self.x)
            # 需要更新的个体
            update_id = np.greater(self.individual_best_fitness,fitness)
            self.p[update_id] = self.x[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id]
            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
            if np.min(fitness) < self.global_best_fitness:
                self.pg = self.x[np.argmin(fitness)]
                self.global_best_fitness = np.min(fitness)
                self.best_x = np.copy(self.x[np.argmin(fitness)])
            self.all_fitness.append(self.global_best_fitness)

        x_label = np.arange(0, len(self.all_fitness), 1)
        plt.title('pso')
        plt.plot(x_label, self.all_fitness, color='blue')
        plt.xlabel('iteration')
        plt.ylabel('fx')
        plt.savefig('.'+'\\'+'PSO_result.png')
        plt.show()

def saveData(path, data):
    with open(path, 'w+', errors='ignore', newline='') as csvfile:
        write = csv.writer(csvfile)
        for i in data:
            write.writerow(i)

def solve(dim=20):
    global value_down_range, value_up_range
    value_up_range = 200
    value_down_range = -200

    t1 = time.perf_counter()
    pso = PSO(dim, 20, 500)
    pso.evolve()
    t2 = time.perf_counter()
    print('X_best Y: ', object_function(pso.best_x))
    print(f'耗时：{t2-t1}')


if __name__ == '__main__':
    solve()
