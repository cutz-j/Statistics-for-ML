import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, cdist
import time

ratings = pd.read_csv("c:/data/ratings.csv")
print(ratings.head())

movies = pd.read_csv("c:/data/movies.csv")
print(movies.head())

ratings = pd.merge(ratings[['userId', 'movieId', 'rating']],
                   movies[['movieId', 'title']], how='left', left_on='movieId',
                   right_on='movieId')

rp = ratings.pivot_table(columns=['movieId'], index=['userId'], values='rating')
rp = rp.fillna(0)
rp_mat = rp.as_matrix()


## 사용자-사용자 유사도 행렬 ## --> 코사인 유사도
m, n = rp.shape
mat_users = np.zeros(shape=(m, m))
for i in range(m):
    for j in range(m):
        if i != j:
            mat_users[i][j] = (1 - cosine(rp_mat[i,:], rp_mat[j,:]))
pd_users = pd.DataFrame(mat_users, index=rp.index, columns=rp.index)
            
def top_sim(uid=16, n=5):
    ## 유사도 높은 사람 출력 ##
    users = pd_users.loc[uid, :].sort_values(ascending=False)
    top_users = users.iloc[:n,]
    top_users = top_users.rename('score')
    print("sim: ", uid)
    return pd.DataFrame(top_users)

print(top_sim(uid=17, n=10))

def top_ratings(uid=16, n_ratings=10):
    ## 해당 ID의 top ratings 영화 출력 ##
    uid_ratings = ratings.loc[ratings['userId'] == uid]
    uid_ratings = uid_ratings.sort_values(by='rating', ascending=False)
    print("Top: ", n_ratings, "movie ratings: ", uid)
    return uid_ratings.iloc[:n_ratings]

print(top_ratings(uid=17))

start_time = time.time()
mat_movies = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j:
            mat_movies[i][j] = (1 - cosine(rp_mat[:, i], rp_mat[:, j]))
print("%s seconds" % (time.time() - start_time))

pd_movies = pd.DataFrame(mat_movies, index=rp.columns, columns=rp.columns)

def top_movies(mid=588, n=15):
    ## 평점 기반, 유사 영화 n개 출력 ##
    mid_ratings = pd_movies.loc[mid, :].sort_values(ascending=False)
    top_movies = pd.DataFrame(mid_ratings.iloc[:n,])
    top_movies['index1'] = top_movies.index
    top_movies['index1'] = top_movies['index1'].astype('int64')
    top_movies = pd.merge(top_movies, movies[['movieId', 'title']],
                          how='left', left_on='index1', right_on='movieId')
    print("movie sim id: ", mid, movies['title'][movies['movieId']==mid].to_string(index=False),",are")
    del top_movies['index1']
    return top_movies

print(top_movies(mid=589, n=15))
            
### ALS를 사용한 협업 필터링 ###
A = rp.values
W =  A >= 0.5
W[W==True] = 1
W[W==False] == 0
W = W.astype(np.float64, copy=False)
            
W_pred =  A < 0.5
W_pred[W_pred==True] = 1
W_pred[W_pred==False] == 0
W_pred = W_pred.astype(np.float64, copy=False)
np.fill_diagonal(W_pred, val=0)        

## parameters ##
m, n = A.shape
n_iteration = 200
k = 100
lmbda = 0.1

# init #
X = 5 * np.random.rand(m, k)
Y = 5 * np.random.rand(k, n)

def error(A, X, Y, W):
    ## RMSE 계산 ##
    return np.sqrt(np.sum((W * (A - np.dot(X, Y)))**2)/ np.sum(W))

## 학습 ! ##
errors = []
for itr in range(n_iteration):
    X = np.linalg.solve(np.dot(Y, Y.T) + lmbda * np.eye(k), np.dot(Y, A.T)).T
    Y = np.linalg.solve(np.dot(X.T, X) + lmbda * np.eye(k), np.dot(X.T, A))
    if itr % 10 == 0:
        print("error: ", error(A, X, Y, W))
    errors.append(error(A, X, Y, W))
    
## 예측 ##
A_hat = np.dot(X, Y)

# 추천 #
def print_recom(uid=315, n_movies=15, pred_mat=A_hat, wpred_mat=W_pred):
    ## 상위 영화 추천 ##
    pred_recos = pred_mat * wpred_mat
    pd_predrecos = pd.DataFrame(pred_recos, index=rp.index, columns=rp.columns)
    pred_ratings = pd_predrecos.loc[uid, :].sort_values(ascending=False)
    pred_topratings = pred_ratings[:n_movies,]
    pred_topratings = pred_topratings.rename('pred_ratings')
    pred_topratings = pd.DataFrame(pred_topratings)
    pred_topratings['index1'] = pred_topratings.index
    pred_topratings['index1'] = pred_topratings['index1'].astype('int64')
    pred_topratings = pd.merge(pred_topratings, movies[['movieId', 'title']],
                               how='left', left_on='index1', right_on='movieId')
    del pred_topratings['index1']
    print("\ntop", n_movies, "movies predicted for the user: ", uid,
           " based on cf\n")
    return pred_topratings

predmtrx = print_recom(n_movies=10)
print(predmtrx)
























 
            
            
        