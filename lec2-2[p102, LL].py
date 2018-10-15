from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

wine_quality = pd.read_csv("d:/data/wine/winequality-red.csv", sep=';')
wine_quality.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)
columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
           'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 
           'pH','sulphates', 'alcohol']

pdx = wine_quality[columns]
pdy = wine_quality['quality']

x_train, x_test, y_train, y_test = train_test_split(pdx, pdy, train_size=0.7, random_state=77)

alphas = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0]
initrsq = 0
for alph in alphas:
    ridge_reg = Ridge(alpha=alph)
    ridge_reg.fit(x_train, y_train)
    tr_rsqrd = ridge_reg.score(x_train,y_train)
    ts_rsqrd = ridge_reg.score(x_test, y_test)
    if ts_rsqrd > initrsq:
        print("lambda: ", alph, "train r-sq: ", round(tr_rsqrd, 5), "test r-sq :", round(ts_rsqrd, 5))
        initrsq = ts_rsqrd

initrsq = 0        
for alph in alphas:
    lasso_reg = Lasso(alpha=alph)
    lasso_reg.fit(x_train, y_train)
    tr_rsqrd = lasso_reg.score(x_train, y_train)
    ts_rsqrd = lasso_reg.score(x_test, y_test)
    
    if ts_rsqrd > initrsq:
        print("lambda: ", alph, "R-sq ", round(tr_rsqrd, 5), "r-sq ", round(ts_rsqrd, 5))
        initrsq = ts_rsqrd