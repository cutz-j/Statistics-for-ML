import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

random.seed(777)

def random_point_gen(dimension):
    '''
    정규분포 0 ~ 1 사이의 난수 발생 (차원의 수 만큼 반복해서 리스트에 comprehension)
    '''
    return [random.random() for _ in range(dimension)]

def distance(v, w):
    '''
    유클리디언 거리 제곱 -> sum of sq, sqrt 계산
    두 점간의 거리는 좌표끼리의 차이를 각각 제곱해 더하고, sqrt
    '''
    vec_sub = [v_i - w_i for v_i, w_i in zip(v, w)] # 두 벡터의 차이
    sum_of_sq = sum(v_i * v_i for v_i in vec_sub) # 거리 제곱
    return math.sqrt(sum_of_sq) # 거리 제곱근 리턴

def minkowski(v, w, p=2):
    '''
    minkowski, p
    '''
    vec_sub = [np.abs(v_i - w_i) ** p for v_i, w_i in zip(v,w)]
    sum_of_sq = sum(v_i for v_i in vec_sub) ** (1/p)
    return sum_of_sq
    
def random_dist_comparison(dimension, number_pairs):
    '''
    random 정규 분포 2개의 vector 거리 계산
    '''
    return [minkowski(random_point_gen(dimension), random_point_gen(dimension), p=1) for _ in range(number_pairs)]

def mean(x):
    return sum(x) / len(x)

dimensions = range(1, 201, 5) # dim 5차원부터, 200차원까지

## 최소 & 평균거리 계산 ##
avg_dist = []
min_dist = []

dummyarray = np.empty((20,4))
dist_vals = pd.DataFrame(dummyarray)
dist_vals.columns = ["Dimension", "Min_Distance", "Avg_Distance", "Min/Avg_Distance"]

i = 0
for dims in dimensions:
    distances = random_dist_comparison(dims, 100) # 100쌍의 차원마다 계산 1 ~ 196 5단위
    avg_dist.append(mean(distances)) # 평균
    min_dist.append(min(distances)) # 최소
    dist_vals.loc[i, "Dimension"] = dims # df에 기록
    dist_vals.loc[i, "Min_Distance"] = min(distances)
    dist_vals.loc[i, "Avg_Distance"] = mean(distances)
    dist_vals.loc[i, "Min/Avg_Distance"] = min(distances) / mean(distances)
    i += 1


# graph
plt.figure()
plt.xlabel("dim")
plt.ylabel("dist")
plt.plot(dist_vals["Dimension"], dist_vals["Avg_Distance"], "r-")
plt.legend(loc='best')
plt.show()

## 2차원 ,3차원 ##
two = np.random.rand(60, 2)
two2 = np.random.rand(60, 2)
two_df = pd.DataFrame(two)
two_df.columns = ["x", "y"]

#plt.figure()
#plt.scatter(two_df['x'], two_df['y'])
#plt.show()

three = np.random.rand(60, 3)
three2 = np.random.rand(60, 3)
three_df = pd.DataFrame(three)
three_df.columns = ['x', 'y', 'z']
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(three_df['x'], three_df['y'], three_df['z'])
#plt.show()
#print(np.sqrt(np.sum(np.square(two - two2)))) # 4.7
#print(np.sqrt(np.sum(np.square(three - three2)))) # 5.7










































