import numpy as np
from scipy import stats
import pandas as pd

xbar = 990
h0 = 1000
s = 12.5
n = 30

st = (xbar - h0) / (s / np.sqrt(float(n)))
#print(st)

# t-table
alpha = 0.05
t_alpha = stats.t.ppf(alpha, n-1) # 신뢰수준, df
#print(t_alpha) # 임계치보다 검정 통계량이 작다 -> 기각

# P-VALUE
p_val = stats.t.sf(np.abs(st), n-1)
#print(p_val) # 0.05 > 0.0007 --> 기각

## X2 ##
survey = pd.read_csv("Chapter01/survey.csv")
survey_tab = pd.crosstab(survey.Smoke, survey.Exer, margins=True)
observed = survey_tab.ix[:-1, :-1]
contg = stats.chi2_contingency(observed=observed)
p_value = round(contg[1], 3)
# p_value = 0.483, 차이가 없다

fet = pd.read_csv("Chapter01/fetilizers.csv")
anova = stats.f_oneway(fet.fertilizer1, fet.fertilizer2, fet.fertilizer3)
# F_onewayResult(statistic=3.6634935025687523, pvalue=0.05063590143901569)
# 기각 X // 세 집단 중 어느 집단도 차이가 보이지 않는다.