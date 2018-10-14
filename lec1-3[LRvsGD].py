import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

train_data = pd.read_csv("Chapter01/mtcars.csv")

X = np.array(train_data.hp).reshape(32,1)
y = np.array(train_data.mpg).reshape(32,1)

model = LinearRegression(fit_intercept=True)
model.fit(X, y)

print(model.intercept_[0], model.coef_[0])

def gradient_descent(x, y, learning_rate, conv_threshold, batch_size, max_iter):
    converged = False
    iteration = 0
    m = batch_size
    t0 = np.random.random(x.shape[1])
    t1 = np.random.random(x.shape[1])
    MSE = (sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)])) / m
    while not converged:
        grad0 = 1.0 / m * sum([(t0 + t1*x[i] - y[i]) for i in range(m)]) # theta0 미분
        grad1 = 1.0 / m * sum([(t0 + t1*x[i] - y[i]) * x[i] for i in range(m)]) # theta1 미분
        temp0 = t0 - learning_rate * grad0
        temp1 = t1 - learning_rate * grad1
        t0 = temp0
        t1 = temp1
        MSE_new = (sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)])) / m
        if iteration % 200 == 0:
            print(MSE_new)
        if abs(MSE - MSE_new) <= conv_threshold:
            print(iteration)
            converged = True
        MSE = MSE_new
        iteration += 1
        if iteration == max_iter:
            converged = True
    return t0, t1

# main
Inter, Coeff = gradient_descent(x = X,y = y, learning_rate=0.00003,conv_threshold=1e-8, batch_size=32,max_iter=1500000)
print(Inter, Coeff)