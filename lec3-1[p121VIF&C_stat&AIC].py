import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc


colList = ['Status of existing checking account ', 'Duration in month', 'Credit history',
           'Purpose', 'Credit amount', 'Savings Account', 'Present Employment since', 'Installment rate in percentage of disposable income',
           'Personal status and sex', 'Other debtors', 'Present residence since',
           'Property', 'Age in years', 'Other installment plans', 'Housing',
           'Number of existing credits at this bank', 'Job', 'Number of people being liable to provide maintenance for',
           'Telephone', 'Foreign worker', 'class']

for i in range(len(colList)):
    colList[i] = colList[i].replace(" ", "_")
    if colList[i][-1] == "_":
        colList[i] = colList[i][:-1]
    
os.chdir("d:/data/ML") # working directory 변경
credit_data = pd.read_csv("german.data.txt", sep=' ', header=None) # text파일도 read_csv 가능
# sep은 columns 분리, header 없이
credit_data.columns = colList
#print(credit_data.head(5)) # head 요약

credit_data['class'] = credit_data['class'] - 1
#print(credit_data.head(5)) # head 요약

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
        dataf = data.groupby(['bin_var'])['class'].agg(['count','sum'])
        dataf.columns = ["Total", "bad"]
        dataf["good"] = dataf["Total"] - dataf["bad"]
        dataf["bad_per"] = dataf["bad"] / dataf["bad"].sum()
        dataf["good_per"] = dataf["good"] / dataf["good"].sum()
        dataf["I_V"] = (dataf["good_per"] - dataf["bad_per"]) * np.log(dataf["good_per"] / dataf["bad_per"])
    return dataf

#print(IV_calc(credit_data, 'Credit_history'))
#print(IV_calc(credit_data, 'Duration_in_month'))

## IV_list sort ##
ivList = []
for col in credit_data.columns:
    assigned_data = IV_calc(credit_data, var=col)
    iv_val = round(assigned_data["I_V"].sum(), 3)
    dt_type = credit_data[col].dtypes
    ivList.append((iv_val, col, dt_type))

ivList = sorted(ivList, reverse=True)
ivList = ivList[1:]

# Retaining top 15 variables
dummy_stseca = pd.get_dummies(credit_data['Status_of_existing_checking_account'], prefix='status_exs_accnt')
dummy_ch = pd.get_dummies(credit_data['Credit_history'], prefix='cred_hist')
dummy_purpose = pd.get_dummies(credit_data['Purpose'], prefix='purpose')
dummy_savacc = pd.get_dummies(credit_data['Savings_Account'], prefix='sav_acc')
dummy_presc = pd.get_dummies(credit_data['Present_Employment_since'], prefix='pre_emp_snc')
dummy_perssx = pd.get_dummies(credit_data['Personal_status_and_sex'], prefix='per_stat_sx')
dummy_othdts = pd.get_dummies(credit_data['Other_debtors'], prefix='oth_debtors')


dummy_property = pd.get_dummies(credit_data['Property'], prefix='property')
dummy_othinstpln = pd.get_dummies(credit_data['Other_installment_plans'], prefix='oth_inst_pln')
dummy_forgnwrkr = pd.get_dummies(credit_data['Foreign_worker'], prefix='forgn_wrkr')

#dummy_housing = pd.get_dummies(credit_data['Housing'], prefix='housing')
#dummy_job = pd.get_dummies(credit_data['Job'], prefix='job')
#dummy_telephn = pd.get_dummies(credit_data['Telephone'], prefix='telephn')


continuous_columns = ['Duration_in_month', 'Credit_amount','Installment_rate_in_percentage_of_disposable_income',
                       'Age_in_years','Number_of_existing_credits_at_this_bank' ]


credit_continuous = credit_data[continuous_columns]
credit_data_new = pd.concat([dummy_stseca,dummy_ch,dummy_purpose,dummy_savacc,dummy_presc,dummy_perssx,
                             dummy_property,dummy_othinstpln,dummy_othdts,
                             dummy_forgnwrkr,credit_continuous,credit_data['class']],axis=1)

x_train, x_test, y_train, y_test = train_test_split(credit_data_new.drop(['class'], axis=1), 
                                                    credit_data_new['class'], train_size=0.7, random_state=77)

x_train = pd.DataFrame(x_train)
y_train = pd.DataFrame(y_train)

remove_cols_extra_dummy = []
tmp2 = None
for col in x_train.columns:
    if "_A" in col:
        tmp = col[:col.index("A")]
        if tmp != tmp2:
            remove_cols_extra_dummy.append(col)
            tmp2 = col[:col.index("A")]

## 무의미 dummy 변수와 다중공선성 변수를 list에 모아 훈련 때 제거 ##
remove_cols_insig = []
remove_cols = list(set(remove_cols_extra_dummy + remove_cols_insig))

## Logitstic 함수 & n 변수(?) ##

## 다중공선성 VIF ##
def VIF(x_train, remove_cols):
    vifList = []
    cnames = x_train.drop(remove_cols, axis=1).columns
    for i in np.arange(0, len(cnames)):
        xvars = list(cnames)
        yvar = xvars.pop(i)
        mod = sm.OLS(x_train.drop(remove_cols, axis=1)[yvar], sm.add_constant(x_train.drop(remove_cols, axis=1)[xvars]))
        res = mod.fit()
        vif = 1 / (1 - res.rsquared)
        vifList.append((yvar, round(vif, 3)))
    vifList = sorted(vifList, key=lambda a: a[1], reverse=True)
    return vifList
    
## C-통계량 ##
def c_stat(x_train, y_train, remove_cols):
    global both
    y_pred = pd.DataFrame(logistic_model.predict(sm.add_constant(x_train.drop(remove_cols,axis=1))))
    y_pred.columns = ["probs"]
    both = pd.concat([y_train, y_pred], axis=1)
    
    zeros = both[['class', 'probs']][both['class']==0]
    ones = both[['class', 'probs']][both['class']==1]
    
    def df_crossjoin(df1, df2, **kwargs):
        df1['_tmpkey'] = 1
        df2['_tmpkey'] = 1
        res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
        res.index = pd.MultiIndex.from_product((df1.index, df2.index))
        df1.drop('_tmpkey', axis=1, inplace=True)
        df2.drop('_tmpkey', axis=1, inplace=True)
        return res
    
    # 일치쌍 / 불일치쌍 / 동일쌍 분류 (one hot)
    joined_data = df_crossjoin(ones, zeros)
    joined_data['concordant_pair'] = 0
    joined_data.loc[joined_data['probs_x'] > joined_data['probs_y'], 'concordant_pair'] = 1
    joined_data['discordant_pair'] = 0
    joined_data.loc[joined_data['probs_x'] < joined_data['probs_y'], 'discordant_pair'] = 1
    joined_data['tied_pair'] = 0
    joined_data.loc[joined_data['probs_x'] == joined_data['probs_y'], 'tied_pair'] = 1
    
    p_conc = (sum(joined_data['concordant_pair']) * 1.0) / (joined_data.shape[0])
    p_disc = (sum(joined_data['discordant_pair']) * 1.0) / (joined_data.shape[0])
    
    c_stat = 0.5 + (p_conc - p_disc) / 2.0
    return c_stat

#print(c_stat) # 0.8337 --> 83% / AIC(-315.09)
# oth_inst_pln_A142 : p-value(0.995) / VIF(1.365)

#logistic_model = sm.Logit(y_train, sm.add_constant(x_train.drop(remove_cols, axis=1))).fit()
#print(logistic_model.summary()) # AIC: -315.09
#print(VIF(x_train, remove_cols))
#print(c_stat) # 83%
    
remove_cols.append('oth_inst_pln_A142') # p: 0.995, vif: 1.365
#logistic_model = sm.Logit(y_train, sm.add_constant(x_train.drop(remove_cols, axis=1))).fit()
#print(logistic_model.summary()) # AIC: -315.09
#print(VIF(x_train, remove_cols))
#print(c_stat(x_train, y_train, remove_cols)) # 83%
    
remove_cols.append('status_exs_accnt_A12') # p:0.815, vif: 1.725
#logistic_model = sm.Logit(y_train, sm.add_constant(x_train.drop(remove_cols, axis=1))).fit()
#print(logistic_model.summary()) # AIC: -315.09
#print(VIF(x_train, remove_cols))
#print(c_stat(x_train, y_train, remove_cols)) # 83%
    
remove_cols.append('purpose_A46') # p:0.815, vif: 1.725
#logistic_model = sm.Logit(y_train, sm.add_constant(x_train.drop(remove_cols, axis=1))).fit()
#print(logistic_model.summary()) # AIC: -315.13
#print(VIF(x_train, remove_cols))
#print(c_stat(x_train, y_train, remove_cols)) # 83%
    
remove_cols.append('cred_hist_A32') # p:0.351, vif: 7.512
#logistic_model = sm.Logit(y_train, sm.add_constant(x_train.drop(remove_cols, axis=1))).fit()
#print(logistic_model.summary()) # AIC: -315.57
#print(VIF(x_train, remove_cols))
#print(c_stat(x_train, y_train, remove_cols)) # 83%

remove_cols.append('per_stat_sx_A93') # p:0.128, vif: 6.189
#logistic_model = sm.Logit(y_train, sm.add_constant(x_train.drop(remove_cols, axis=1))).fit()
#print(logistic_model.summary()) # AIC: -316.72
#print(VIF(x_train, remove_cols))
#print(c_stat(x_train, y_train, remove_cols)) # 83%

remove_cols.append('purpose_A48') # p:1.000, vif: 1.077
#logistic_model = sm.Logit(y_train, sm.add_constant(x_train.drop(remove_cols, axis=1))).fit()
#print(logistic_model.summary()) # AIC: -320.52
#print(VIF(x_train, remove_cols))
#print(c_stat(x_train, y_train, remove_cols)) # 82%

remove_cols.append('pre_emp_snc_A75') # p:0.882, vif: 3.88
remove_cols.append('pre_emp_snc_A73') # p:0.794, vif: 4.209
remove_cols.append('pre_emp_snc_A72') # p:0.714, vif: 3.199
remove_cols.append('pre_emp_snc_A74') # p:0.086, vif: 3.093
#logistic_model = sm.Logit(y_train, sm.add_constant(x_train.drop(remove_cols, axis=1))).fit()
#print(logistic_model.summary()) # AIC: -323.45
#print(VIF(x_train, remove_cols))
#print(c_stat(x_train, y_train, remove_cols)) # 82%

remove_cols.append('cred_hist_A33') # p:0.554, vif: 1.194
remove_cols.append('property_A123') # p:0.486, vif: 1.641
remove_cols.append('purpose_A44') # p:0.710, vif: 1.059
remove_cols.append('oth_debtors_A103') # p:0.086, vif: 1.122
#logistic_model = sm.Logit(y_train, sm.add_constant(x_train.drop(remove_cols, axis=1))).fit()
#print(logistic_model.summary()) # AIC: -324.47
#print(VIF(x_train, remove_cols))
#print(c_stat(x_train, y_train, remove_cols)) # 82%

remove_cols.append('property_A122')
remove_cols.append('purpose_A45')
remove_cols.append('sav_acc_A62')
remove_cols.append('status_exs_accnt_A13')
#logistic_model = sm.Logit(y_train, sm.add_constant(x_train.drop(remove_cols, axis=1))).fit()
#print(logistic_model.summary()) # AIC: -326.76
#print(VIF(x_train, remove_cols))
#print(c_stat(x_train, y_train, remove_cols)) # 81%

remove_cols.append('cred_hist_A31')
remove_cols.append('purpose_A49')
remove_cols.append('sav_acc_A63')
remove_cols.append('forgn_wrkr_A202')
#logistic_model = sm.Logit(y_train, sm.add_constant(x_train.drop(remove_cols, axis=1))).fit()
#print(logistic_model.summary()) # AIC: -331
#print(VIF(x_train, remove_cols))
#print(c_stat(x_train, y_train, remove_cols)) # 81%

remove_cols.append('purpose_A42')
remove_cols.append('sav_acc_A64')
remove_cols.append('per_stat_sx_A94')
remove_cols.append('Age_in_years')
#logistic_model = sm.Logit(y_train, sm.add_constant(x_train.drop(remove_cols, axis=1))).fit()
#print(logistic_model.summary()) # AIC: -337.70
#print(VIF(x_train, remove_cols))
#print(c_stat(x_train, y_train, remove_cols)) # 80%

remove_cols.append('purpose_A410')
remove_cols.append('per_stat_sx_A92')
remove_cols.append('property_A124')
remove_cols.append('Number_of_existing_credits_at_this_bank')
logistic_model = sm.Logit(y_train, sm.add_constant(x_train.drop(remove_cols, axis=1))).fit()
print(logistic_model.summary()) # AIC: -343
print(VIF(x_train, remove_cols))
print(c_stat(x_train, y_train, remove_cols)) # 79%

fpr, tpr, thresholds = metrics.roc_curve(both['class'], both['probs'], pos_label=1)
roc_auc = auc(fpr, tpr)
#plt.figure()
lw = 2
#plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area=%0.2f)' % roc_auc)
#plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.legend(loc="lower right")
#plt.show()

## 임계치 그리드 검색 ##
for i in list(np.arange(0,1,0.1)):
    both["y_pred"] = 0
    both.loc[both["probs"] > i, 'y_pred'] = 1
    print("Trheshold", i, "Train Accruacy: ", round(metrics.accuracy_score(both['class'], both['y_pred']), 4))

## 혼동 행렬 생성 (0.5) ##
def ctr_value(data, alpha):    
    global both
    data["y_pred"] = 0
    data.loc[data["probs"] > alpha, 'y_pred'] = 1
    print("\nTrain Confusion Matrix\n\n", pd.crosstab(data['class'],
                                                      data['y_pred'], rownames=["Actual"], 
                                                      colnames=["Predicted"]))
    return metrics.accuracy_score(data['class'], data['y_pred'])

ctr_value(both, 0.5)

y_pred_test = pd.DataFrame(logistic_model.predict(sm.add_constant(x_test.drop(remove_cols, axis=1))))
y_pred_test.columns = ["probs"]
both_test = pd.concat([y_test, y_pred_test], axis=1)
acc = ctr_value(both_test, 0.5)
print(acc) # 73%






        