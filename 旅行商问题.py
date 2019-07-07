# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:40:56 2019

@author: morichina
"""

import numpy as np
import matplotlib.pyplot as plt

cities = np.array([[520.0,585.0],[480.0,415.0],[835.0,625.0],[975.0,580.0],[1215.0,245.0],
          [1320.0,315.0],[1250.0,400.0],[660.0,180.0],[410.0,250.0],[420.0,555.0]])
def getDistmat(cities):
    num = cities.shape[0]
    distmat = np.zeros((num, num))
    for i in range(num):
        for j in range(i, num):
            distmat[i][j] = distmat[j][i] = np.linalg.norm(cities[i] - cities[j])
    return distmat

distmat = getDistmat(cities)

numAnt = 15
numCity = cities.shape[0]
alpha = 2 # 信息素重要程度
belta = 5 # 启发因子重要程度
rho = 0.1 # 信息素挥发速率
Q = 1

iter = 0
iterMax = 100

lengthBest = np.zeros(iterMax) # 记录每次迭代的最短路径长度
pathBest = np.zeros((iterMax, numCity)).astype(int) #记录每次迭代的最短路径

heuristic = Q/(distmat + np.diag([1e10]*numCity)) # 启发因子矩阵
pheromone = np.ones((numCity, numCity)) # 信息素矩阵

while iter < iterMax:
    
    length = np.zeros(numAnt) # 记录每个蚂蚁走过的路径长度
    path = np.zeros((numAnt, numCity)) # 记录每个蚂蚁的路径
    path[0:numCity, 0] = np.random.permutation(range(numCity))
    path[numCity: , 0] = np.random.permutation(range(numCity))[0:numAnt - numCity]
    
    for i in range(numAnt):
        unVisited = set(range(numCity))
        visiting = path[i, 0]
        unVisited.remove(visiting)
        for j in range(1, numCity):
            listUnVisited = list(unVisited)
            probablity = np.zeros(len(listUnVisited))
            for k in range(len(listUnVisited)):
                probablity[k] = np.power(pheromone[int(visiting)][int(listUnVisited[k])], alpha)\
                *np.power(heuristic[int(visiting)][int(listUnVisited[k])], belta)
            newProbablity = probablity/sum(probablity)
            cumProbablity = newProbablity.cumsum() - np.random.rand()
            nextCity = listUnVisited[int(np.where(cumProbablity > 0)[0][0])]
            length[i] += distmat[int(visiting)][int(nextCity)]
            path[i][j] = nextCity
            visiting = nextCity
            unVisited.remove(visiting)
        length[i] += distmat[int(visiting)][int(path[i][0])]
        
    changePheromone = np.zeros((numCity, numCity))
    for i in range(numAnt):
        for j in range(numCity - 1):
            changePheromone[int(path[i][j])][int(path[i][j+1])] += Q/distmat[int(path[i][j])][int(path[i][j+1])]
            changePheromone[int(path[i][j+1])][int(path[i][j])] += Q/distmat[int(path[i][j])][int(path[i][j+1])]
        changePheromone[int(path[i][numCity - 1])][int(path[i][0])] += Q/distmat[int(path[i][numCity -1])][int(path[i][0])]
        changePheromone[int(path[i][0])][int(path[i][numCity - 1])] += Q/distmat[int(path[i][numCity -1])][int(path[i][0])]
    pheromone = (1 - rho) * pheromone + changePheromone
            
    lengthBest[iter] = length.min()
    pathBest[iter] = path[length.argmin()].copy()
    iter = iter + 1
    if iter % 10 == 0:
        print(iter)
minLength = lengthBest.min()
print("路径最小长度为{}".format(minLength))
print("最短路径如下")
bestPath = pathBest[lengthBest.argmin()].copy()

plt.plot(cities[:,0],cities[:,1],'r.',marker=u'$\cdot$')
plt.xlim([0, 2000])
plt.ylim([0, 1500])  

for i in range(numCity-1):#
    m,n = bestPath[i],bestPath[i+1]
    print(m,n)
    m = int(m)
    n = int(n)
    plt.plot([cities[m][0],cities[n][0]],[cities[m][1],cities[n][1]],'k')
plt.plot([cities[bestPath[0]][0],cities[int(n)][0]],[cities[bestPath[0]][1],cities[int(n)][1]],'b')
plt.show()          