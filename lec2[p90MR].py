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
sns.set(style='whitegrid', context='notebook')
#sns.pairplot(wine_quality[eda_columns], size=2.5, x_vars=eda_columns, y_vars=eda_columns)
#plt.show()

wine_sel = wine_quality[eda_columns]
corr_mat = np.corrcoef(wine_sel.values.T)
sns.set(font_scale=1)
#full_mat = sns.heatmap(corr_mat, cbar=True, annot=True, square=True, fmt='.2f',
#                       annot_kws={'size': 15}, yticklabels=eda_columns, xticklabels=eda_columns)
#plt.show()