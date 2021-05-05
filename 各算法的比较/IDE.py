"""
IDE算法：
测试函数：
Griewank函数
f = 1 + (add_all(x[i]**2))/4000 - multiply_all(math.cos(x[i])/(math.sqrt(i)))

"""
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt


#处理Griewank函数
def add_all(x):
    sum_all = 0
    for i in range(len(x)):
        sum_all += x[i]**2
    return sum_all


def multiply_all(x):
    if len(x) == 0:
        return 1

    multi_sum = 0
    for i in range(0,len(x)):
        multi_sum *= math.cos(x[i]) / math.sqrt(i+0.0001)  #避免除以0
    return multi_sum


def object_func(x):
    return 1 + add_all(x)/4000 - multiply_all(x)


# 防止越界
def controlRange(x):
    global value_up_range, value_down_range
    if x > value_up_range:
        while x > value_up_range:
            x -= value_up_range - value_down_range
    elif x <= value_down_range:
        while x < value_down_range:
            x += value_up_range - value_down_range
    return x


# 列表相减
def substract(a_list, b_list):
    global value_up_range, value_down_range
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        new_list.append(controlRange(a_list[i] - b_list[i]))
    return new_list


# 列表相加
def add(a_list, b_list):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        new_list.append(controlRange(a_list[i] + b_list[i]))
    return new_list


# 列表的数乘
def multiply(a, b_list):
    b = len(b_list)
    new_list = []
    for i in range(0, b):
        new_list.append(a * b_list[i])
    return new_list


#种群初始化
def initiate(NP, x_len, value_up_range, value_down_range):
    np_list = []
    for i in range(0, NP):
        x_list = [] #个体，基因数x_len
        for j in range(0, x_len):
            x_list.append(value_down_range + random.random() * (value_up_range-value_down_range))
        np_list.append(x_list)
    return np_list


#变异
def mutation(NP, np_list):
    global x_best, y_best
    v_list = []
    for i in range(0,NP):
        r1 = random.randint(0, NP - 1)
        while r1 == i:
            r1 = random.randint(0, NP - 1)
        r2 = random.randint(0, NP - 1)
        while r2 == r1 | r2 == i:
            r2 = random.randint(0, NP - 1)
        r3 = random.randint(0, NP - 1)
        while r3 == r2 | r3 == r1 | r3 == i:
            r3 = random.randint(0, NP - 1)

        #IDE的变异
        t1 = multiply(F, substract(np_list[r2], np_list[r3]))
        t2 = multiply(F, substract(x_best, np_list[r1]))
        t3 = x_best
        t = add(add(t1, t2), t3)
        v_list.append(t)

    return v_list


#交叉
def crossover(np_list, v_list):
    u_list = []
    for i in range(0, NP):
        vv_list = []
        for j in range(0, x_len):
            # ide与de的交换方法相同
            if (random.random() <= CR) | (j == random.randint(0, x_len - 1)):
                vv_list.append(v_list[i][j])
            else:
                vv_list.append(np_list[i][j])

        u_list.append(vv_list)
    return u_list


#选择
def selection(u_list, np_list):
    global x_best, y_best, not_progress_step
    exitBetterFit = False
    for i in range(0, NP):
        ui_y = object_func(u_list[i])
        npi_y = object_func(np_list[i])
        if ui_y <= npi_y:
            np_list[i] = u_list[i]
            if ui_y < y_best:
                exitBetterFit = True
                y_best = ui_y
                x_best = [j for j in u_list[i]]
        else:
            np_list[i] = np_list[i]
    if exitBetterFit:
        not_progress_step = 0
    else:
        not_progress_step += 1
    return np_list

def initpara():
    NP = 5  # 种群数量
    F = 0.5  # 缩放因子
    CR = 0.7  # 交叉概率
    generation = 500  # 遗传代数
    x_len = 20
    value_up_range = 20
    value_down_range = -20
    max_not_progress_step = 200

    return NP, F, CR, generation, x_len, value_up_range, value_down_range, max_not_progress_step

def IDE():
    global NP, F, CR, generation, x_len, value_up_range, value_down_range, x_best, y_best, max_not_progress_step
    not_progress_step = 0
    x_best = None
    y_best = None
    NP, F, CR, generation, x_len, value_up_range, value_down_range, max_not_progress_step = initpara()
    np_list = initiate(NP, x_len, value_up_range,value_down_range)
    min_x = []
    min_f = []
    xx = []
    for i in range(0, NP):
        y = object_func(np_list[i])
        xx.append(y)
        # 初始化 xbest 最佳个体
        if (y_best == None) or (y < y_best):
            # 复制
            y_best = y
            x_best = [j for j in np_list[i]]
    min_f.append(min(xx))
    min_x.append(np_list[xx.index(min(xx))])
    for i in range(0, generation):
        # 根据当前代数 计算 F值   F = ((g_current)/(generation_max-g_current))**2
        # 根据当前代数 计算 CR值   CR=( g_current/generation_max)**2
        F = ((i + 1) / ((generation + 1) - (i + 1))) ** 2
        CR = ((i + 1) / generation) ** 2

        v_list = mutation(NP, np_list)
        u_list = crossover(np_list, v_list)
        np_list = selection(u_list, np_list)
        if not_progress_step >= max_not_progress_step:
            break
        xx = []
        for i in range(0, NP):
            xx.append(object_func(np_list[i]))
        min_f.append(min(xx))
        min_x.append(np_list[xx.index(min(xx))])

    # 输出
    min_ff = min(min_f)
    min_xx = min_x[min_f.index(min_ff)]
    print('the minimum point is x ')
    print(min_xx)
    print('the minimum value is y ')
    print(min_ff)
    print('the min object_func:', object_func(min_xx))

    return min_f

def SaveDate(it,z, time):
    data = list(zip(it,z))
    data.append('elapsed time:' + str(time))

    s = ''
    with open('IDE_result.txt','w+') as w:
        for d in data:
            s += str(d) + '\n'
        w.writelines(s)

def show(min_f):
    # 画图
    x_label = np.arange(0, generation + 1, 1)
    plt.title('ide')
    plt.plot(x_label, min_f, color='blue')
    plt.xlabel('iteration')
    plt.ylabel('fx')
    plt.savefig('.' + '\\' + 'IDE_result.png')
    plt.show()


if __name__ == '__main__':
    NP, F, CR, generation, len_x, value_up_range, value_down_range, max_not_progress_step = (
        None, None, None, None, None, None, None, None)
    np_list = None
    not_progress_step = 0
    t1 = time.perf_counter()
    min_f = IDE()
    t2 = time.perf_counter()
    t = t2-t1
    print(f'耗时：{t}')
    show(min_f)

    x_label = np.arange(0, generation + 1, 1)
    SaveDate(x_label,min_f,t)