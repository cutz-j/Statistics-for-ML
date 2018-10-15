import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

## function
def mean(df):
    return round(np.sum(df) / float(len(df)), 2)

## main
wine_quality = pd.read_csv("d:/data/wine/winequality-red.csv", sep=';')
#print(wine_quality.head(5))

wine_quality.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)

x_train, x_test, y_train, y_test = train_test_split(wine_quality['alcohol'], wine_quality['quality'], 
                                                    train_size=0.7, random_state=77)

x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

alcohol_mean = mean(x_train)
quality_mean = mean(y_train)
alcohol_var = round(np.sum((x_train - alcohol_mean)**2), 2)
quality_var = round(np.sum((y_train - quality_mean)**2), 2)
covar = round(np.sum(np.array(x_train - alcohol_mean) * np.array(y_train - quality_mean)), 2)
b1 = covar / alcohol_var[0]
b0 = quality_mean[0] - b1 * alcohol_mean[0]
#print(b0, b1)

## R-Square
y_pred = b0 + b1 * x_test
y_pred = y_pred.rename(columns={'alcohol':'quality'})
R_sq = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - mean(y_test))**2))
y_err = y_test - y_pred
hypothesis = b0 + b1 * x_train

## Residual plot
plt.scatter(y_pred, y_err)
plt.show()

# scatter
plt.scatter(x_train, y_train)
plt.plot(x_train, hypothesis, 'r')
plt.show()










