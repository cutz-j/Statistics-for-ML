### pipeline - randomforest ###
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import os
import matplotlib.pyplot as plt
import numpy as np

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

# one-hot -1
credit_data['class'] = credit_data['class'] - 1 

dummy_stseca = pd.get_dummies(credit_data['Status_of_existing_checking_account'], prefix='status_exs_accnt')
dummy_ch = pd.get_dummies(credit_data['Credit_history'], prefix='cred_hist')
dummy_purpose = pd.get_dummies(credit_data['Purpose'], prefix='purpose')
dummy_savacc = pd.get_dummies(credit_data['Savings_Account'], prefix='sav_acc')
dummy_presc = pd.get_dummies(credit_data['Present_Employment_since'], prefix='pre_emp_snc')
dummy_perssx = pd.get_dummies(credit_data['Personal_status_and_sex'], prefix='per_stat_sx')
dummy_othdts = pd.get_dummies(credit_data['Other_debtors'], prefix='oth_debtors')
dummy_property = pd.get_dummies(credit_data['Property'], prefix='property')
dummy_othinstpln = pd.get_dummies(credit_data['Other_installment_plans'], prefix='oth_inst_pln')
dummy_housing = pd.get_dummies(credit_data['Housing'], prefix='housing')
dummy_job = pd.get_dummies(credit_data['Job'], prefix='job')
dummy_telephn = pd.get_dummies(credit_data['Telephone'], prefix='telephn')
dummy_forgnwrkr = pd.get_dummies(credit_data['Foreign_worker'], prefix='forgn_wrkr')

continuous_columns = ['Duration_in_month', 'Credit_amount','Installment_rate_in_percentage_of_disposable_income',
                      'Present_residence_since', 'Age_in_years','Number_of_existing_credits_at_this_bank',
                       'Number_of_people_being_liable_to_provide_maintenance_for']
credit_continuous = credit_data[continuous_columns]

# new df --> 범위형, 연속형 feature를 한 df로 #
credit_data_new = pd.concat([dummy_stseca, dummy_ch, dummy_purpose, dummy_savacc, dummy_presc,
                             dummy_perssx, dummy_othdts, dummy_property, dummy_othinstpln,
                             dummy_housing, dummy_job, dummy_telephn, dummy_forgnwrkr, credit_continuous,
                             credit_data['class']], axis=1) # shape(1000, 62)

# split #
x_train, x_test, y_train, y_test = train_test_split(credit_data_new.drop(['class'], axis=1),
                                                    credit_data_new['class'], train_size=0.7, test_size=0.3, random_state=77)



pipeline = Pipeline([('clf', RandomForestClassifier(criterion='gini'))])
parameters = {'clf__n_estimators':(1000, 2000, 3000),
              'clf__max_depth':(100, 200, 300),
              'clf__min_samples_split':(2, 3),
              'clf__min_samples_leaf':(1, 2)}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=5, verbose=1, scoring='accuracy')
grid_search.fit(x_train, y_train)
print('Best Training Score: %0.3f' % grid_search.best_score_)
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))
    
prediction = grid_search.predict(x_test)
print("accuracy: ", round(accuracy_score(y_test, prediction), 4))
print(classification_report(y_test, prediction))
print(pd.crosstab(y_test, prediction, rownames=["actual"], colnames=["predicted"]))

rf_fit = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=300,
                                min_samples_split=3, min_samples_leaf=1)
rf_fit.fit(x_train, y_train)
importances = rf_fit.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_fit.estimators_], axis=0)

indices = np.argsort(importances)[::-1]
colnames = list(x_train.columns)
for f in range(x_train.shape[1]):
    print(indices[f], colnames[indices[f]], round(importances[indices[f]], 4))

plt.figure()
plt.bar(range(x_train.shape[1]), importances[indices], yerr=std[indices], color="r", align="center")
plt.xticks(range(x_train.shape[1]), indices)
plt.xlim([-1, x_train.shape[1]])
plt.show()










