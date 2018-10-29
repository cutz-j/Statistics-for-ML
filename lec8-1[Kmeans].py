### 붓꽃 데이터 K-means ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

## 데이터 전처리 ##
iris = pd.read_csv("d:\\data\iris.csv")
x_iris = iris.iloc[:,:-1]
y_iris = iris.iloc[:, -1]

k_means_fit = KMeans(n_clusters=3, max_iter=300)
k_means_fit.fit(x_iris)

## confusion maxtrix ## --> 실제 비지도 학습에서는 정답label이 존재하지 않는다.
#print(pd.crosstab(y_iris, k_means_fit.labels_,rownames=["Actual"], colnames=["predicted"]))
#print(silhouette_score(x_iris, k_means_fit.labels_, metric="euclidean"))

## 민감도 분석 ##  --> 실루엣 스코어 
for k in range(2, 10): # 2: 0.681, 3: 0.553
    k_means_fitk = KMeans(n_clusters=k, max_iter=300)
    k_means_fitk.fit(x_iris)
#    print("silhouette_score %i: %0.3f" %(k, silhouette_score(x_iris, k_means_fitk.labels_,
#                                                             metric='euclidean')))
    
## 평균분산 엘보 그래프 ##
K = range(1, 10)

KM = [KMeans(n_clusters=k).fit(x_iris) for k in K] # 각 k마다의 KMeans --> obj list
centroids = [k.cluster_centers_ for k in KM] # k클러스터마다 중앙값
D_k = [cdist(x_iris, center, 'euclidean') for center in centroids] # 각 중앙값과의 관측값과의 거리
cIdx = [np.argmin(D, axis=1) for D in D_k] # 관측값마다의 최소값을 인덱스로 --> axis = 1(행마다)(데이터)
dist = [np.min(D, axis=1) for D in D_k]
avgWithinSS = [sum(d) / x_iris.shape[0] for d in dist]

# SS
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(x_iris)**2) / x_iris.shape[0]
bss = tss - wcss

# 엘보 곡선 #
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
plt.grid(True)
plt.xlabel('K num')
plt.ylabel('Avg')
plt.show()







