"""
模拟退火算法：
旅行家问题：
给出N个城市的坐标，找出城市之间的最短路径，并回到最初的起点
"""

import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt
import time

#随机生成n个城市 , 初始温度为T
def creatCitys(n,T):
    #初始化城市坐标
    city = np.random.randint(1,100,size=(n,2))
    return city


#计算城市之间距离   city[n][0] --> 编号为n的city的x坐标  city[n][1] -->y坐标
def citysLength(city):
    sumDistence = 0
    for i in range(len(city) - 1):
        x_dis = city[i][0] - city[i+1][0]
        y_dis = city[i][1] - city[i+1][1]
        distance = math.sqrt(x_dis**2 + y_dis**2)
        sumDistence += distance
    #最后一个城市到起点的距离
    distance = math.sqrt((city[len(city)-1][0] - city[0][0])**2 + (city[len(city)-1][1] - city[0][1])**2)
    sumDistence += distance
    return sumDistence


#随机打乱城市顺序
def disturb(city):
    n = range(len(city))
    c1 , c2 = random.sample(n,2)
    temp_x = city[c1][0]
    temp_y = city[c1][1]
    city[c1][0] = city[c2][0]
    city[c1][1] = city[c2][1]
    city[c2][0] = temp_x
    city[c2][1] = temp_y
    return city


#模拟退火算法    T : 起始温度
def sa(city , T ):
    cost = []
    while  T > 0.001:
        for i in range(50):
            len1 = citysLength(city)
            per_city = copy.deepcopy(city)
            city = disturb(city)
            len2 = citysLength(city)
            delta = len2 - len1
            if delta < 0:
                pass
            else:
                r = random.random()
                #以一定概率接受坏结果
                if math.exp(((-delta)/T)) > r:
                    pass
                else:
                    city = per_city
        T = float(T) * 0.99
        cost.append(citysLength(city))
    return city,cost


#画图
def show(city,cost):
    plt.plot(cost)
    plt.savefig('.'+'\\'+'SA_result.jpg')
    plt.show()
    
    x_list = []
    y_list = []
    for i in range(len(city)):
        x_list.append(city[i][0])
        y_list.append(city[i][1])
    #为了形成环路，将起点加入到最后
    # x_list.append(city[0][0])
    # y_list.append(city[0][1])
    plt.plot(x_list,y_list)
    plt.savefig('.'+'\\'+'SA_result_city_map.jpg')
    plt.show()
    print('最短路程：%f' % citysLength(city))


if __name__ == '__main__':
    city = creatCitys(20, 2000)
    t1 = time.perf_counter()
    city, cost = sa(city, 2000)
    t2 = time.perf_counter()
    show(city,cost)
    print('耗时：%f'%(t2-t1))