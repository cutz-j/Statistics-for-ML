import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

colList = ['Status of existing checking account ', 'Duration in month', 'Credit history',
           'Purpose', 'Credit amount', 'Savings account/bonds', 'Present employment since', 'Installment rate in percentage of disposable income',
           'Personal status and sex', 'Other debtors / guarantors', 'Present residence since',
           'Property', 'Age in years', 'Other installment plans', 'Housing',
           'Number of existing credits at this bank', 'Job', 'Number of people being liable to provide maintenance for',
           'Telephone', 'foreign worker', 'class']
os.chdir("d:/data/ML") # working directory 변경
credit_data = pd.read_csv("german.data.txt", sep=' ', header=None) # text파일도 read_csv 가능
# sep은 columns 분리, header 없이
credit_data.columns = colList
print(credit_data.head(5)) # head 요약

credit_data['class'] = credit_data['class'] - 1
print(credit_data.head(5)) # head 요약

def IV_calc(data, var):
    '''
    cell value가 글자라면, 범주형 / 숫자라면, 연속형
    '''
    if data[var].dtypes == "object":
        # data의 한 var columns을 'class' 별 (0, 1)로 groupby 후, count와 sum column에 계산치 (agg 함수)
        dataf = data.groupby([var])['class'].agg(['count', 'sum'])
        dataf.columns = ["Total", "bad"] # count(총 횟수)를 Total, 1의 총 합을 bad로
        dataf["good"] = dataf["Total"] - dataf["bad"] # 총 합에서 bad를 뺀 것을 good으로 col 추가
        dataf["bad_per"] = dataf["bad"] / dataf["bad"].sum()
        dataf["good_per"] = dataf["good"] / dataf["good"].sum()
        dataf["I_V"] = (dataf["good_per"] - dataf["bad_per"]) * np.log(dataf["good_per"] / dataf["bad_per"])
        return dataf
    else:
        data['bin_var'] = pd.qcut(data[var].rank(method='first'), 10)
        