### p90 ###

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

wine_quality = pd.read_csv("d:/data/wine/winequality-red.csv", sep=';')
wine_quality.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)
eda_columns = ['volatile_acidity', 'chlorides', 'sulphates', 'alcohol', 'quality']
columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
               'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH',
               'sulphates', 'alcohol']
sns.set(style='whitegrid', context='notebook')
#sns.pairplot(wine_quality[eda_columns], size=2.5, x_vars=eda_columns, y_vars=eda_columns)
#plt.show()

wine_sel = wine_quality[eda_columns]
corr_mat = np.corrcoef(wine_sel.values.T)
sns.set(font_scale=1)
#full_mat = sns.heatmap(corr_mat, cbar=True, annot=True, square=True, fmt='.2f',
#                       annot_kws={'size': 15}, yticklabels=eda_columns, xticklabels=eda_columns)
#plt.show()

# 11개 IV // 1개 DV
pdx = wine_quality[columns]
pdy = wine_quality['quality']

# data split
x_train, x_test, y_train, y_test = train_test_split(pdx, pdy, train_size=0.7, random_state=77)

# add constant // OLS
x_train_new = sm.add_constant(x_train)
x_test_new = sm.add_constant(x_test)
full_mod = sm.OLS(y_train, x_train_new)
full_res = full_mod.fit()
#print(full_res.summary())

#cnames = x_train.columns
#for i in np.arange(0, len(cnames)):
#    xvars = list(cnames)
#    yvar = xvars.pop(i)
#    mod = sm.OLS(x_train[yvar], sm.add_constant(x_train_new[xvars]))
#    res = mod.fit()
#    vif = 1/(1-res.rsquared)
#    print(yvar, round(vif, 3))

###############################################
columns = ['volatile_acidity', 'chlorides',
               'free_sulfur_dioxide', 'total_sulfur_dioxide', 'pH',
               'sulphates', 'alcohol']
pdx = wine_quality[columns]
pdy = wine_quality['quality']
x_train, x_test, y_train, y_test = train_test_split(pdx, pdy, train_size=0.7, random_state=77)
x_train_new = sm.add_constant(x_train)
x_test_new = sm.add_constant(x_test)
full_mod = sm.OLS(y_train, x_train_new)
full_res = full_mod.fit()
print(full_res.summary())

cnames = x_train.columns
for i in np.arange(0, len(cnames)):
    xvars = list(cnames)
    yvar = xvars.pop(i)
    mod = sm.OLS(x_train[yvar], sm.add_constant(x_train_new[xvars]))
    res = mod.fit()
    vif = 1/(1-res.rsquared)
    print(yvar, round(vif, 3))




















