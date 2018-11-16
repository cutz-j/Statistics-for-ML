import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier

## 데이터 전처리 ##
## HR data load ##
hr_data = pd.read_csv("d:/data/WA_Fn-UseC_-HR-Employee-Attrition.csv") # shape 1470, 35

## yes/no -> 1/0 ##
## dummy ## -> 범주형 one-hot
hr_data['Attrition_ind'] = 0
hr_data.loc[hr_data['Attrition'] == 'Yes', 'Attrition_ind'] = 1
dummy_bus = pd.get_dummies(hr_data['BusinessTravel'], prefix='busns_trvl')
dummy_dpt = pd.get_dummies(hr_data['Department'], prefix='dept')
dummy_edufield = pd.get_dummies(hr_data['EducationField'], prefix='edufield')
dummy_gender = pd.get_dummies(hr_data['Gender'], prefix='gend')
dummy_jobrole = pd.get_dummies(hr_data['JobRole'], prefix='jobrole')
dummy_maritstat = pd.get_dummies(hr_data['MaritalStatus'], prefix='maritalstat')
dummy_overtime = pd.get_dummies(hr_data['OverTime'], prefix='overtime')

# cont columns #
continuous_columns = ['Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction',
'HourlyRate', 'JobInvolvement', 'JobLevel','JobSatisfaction','MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction','StockOptionLevel', 'TotalWorkingYears', 
'TrainingTimesLastYear','WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
'YearsWithCurrManager']

hr_continuous = hr_data[continuous_columns]
hr_data_new = pd.concat([dummy_bus, dummy_dpt, dummy_edufield, dummy_gender,
                         dummy_jobrole, dummy_maritstat, dummy_overtime, hr_continuous,
                         hr_data['Attrition_ind']], axis=1)
#hr_data_new.shape # shape (1470, 52)

## split ##
x_train, x_test, y_train, y_test = train_test_split(hr_data_new.drop(['Attrition_ind'], axis=1),
                                                    hr_data['Attrition_ind'], train_size=0.7,
                                                    test_size=0.3, random_state=77)


### decision Tree ###
dt = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2,
                            min_samples_leaf=1, random_state=77)
dt.fit(x_train, y_train)

## confusion_matrix ##
print(pd.crosstab(y_train, dt.predict(x_train), rownames=['actual'], colnames=['predict']))
print(round(accuracy_score(y_train, dt.predict(x_train)), 4))
print(classification_report(y_train, dt.predict(x_train)))

print(pd.crosstab(y_test, dt.predict(x_test), rownames=['actual'], colnames=['predict']))
print(round(accuracy_score(y_test, dt.predict(x_test)), 4))
print(classification_report(y_test, dt.predict(x_test)))

### 가중값 튜닝 ###
dummy = np.empty((6,10))
dt_wttune = pd.DataFrame(dummy)
dt_wttune.columns = ["zero_wght", "one_wght", "tr_accuracy", "tst_accuracy", "prec_zero",
                     "prec_one", "prec_ovil", "recl_zero", "recl_one", "recl_evil"]
zero_clwghts = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
print("=========tuning========")
for i in range(len(zero_clwghts)):
    clwght = {0:zero_clwghts[i], 1:1.0-zero_clwghts[i]}
    dt_fit = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2,
                                    min_samples_leaf=1, random_state=77, class_weight=clwght)
    dt_fit.fit(x_train, y_train)
    dt_wttune.loc[i, 'zero_wght'] = clwght[0]
    dt_wttune.loc[i, 'one_wght'] = clwght[1]
    dt_wttune.loc[i, 'tr_accuracy'] = round(accuracy_score(y_train, dt_fit.predict(x_train)), 3)
    dt_wttune.loc[i ,'tst_accuracy'] = round(accuracy_score(y_test, dt_fit.predict(x_test)), 3)
    clf_sp = classification_report(y_test, dt_fit.predict(x_test)).split()
    dt_wttune.loc[i, 'prec_zero'] = float(clf_sp[5])
    dt_wttune.loc[i, 'prec_one'] = float(clf_sp[10])
    dt_wttune.loc[i, 'prec_ovil'] = float(clf_sp[17])
    dt_wttune.loc[i, 'recl_zero'] = float(clf_sp[6])
    dt_wttune.loc[i, 'recl_one'] = float(clf_sp[11])
    dt_wttune.loc[i, 'recl_ovil'] = float(clf_sp[18])
    print(clwght)
    print(round(accuracy_score(y_train, dt_fit.predict(x_train)), 3), "\t",
          round(accuracy_score(y_test, dt_fit.predict(x_test)), 3))
    print(pd.crosstab(y_test, dt_fit.predict(x_test), rownames=['actual'], colnames=['predict']))
    print()

### 배깅 ###
bag_fit = BaggingClassifier(base_estimator=dt_fit, n_estimators=5000, max_samples=0.67,
                            max_features=1.0, bootstrap=True, bootstrap_features=False,
                            n_jobs=-1, random_state=77)
bag_fit.fit(x_train, y_train)
y_hat = bag_fit.predict(x_train)
print(pd.crosstab(y_train, y_hat, rownames=['actual'], colnames=['predict']))
print(accuracy_score(y_train, y_hat))
print(classification_report(y_train, y_hat))
print()
y_hat_test = bag_fit.predict(x_test)
print(pd.crosstab(y_test, y_hat_test, rownames=['actual'], colnames=['predict']))
print(accuracy_score(y_test, y_hat_test))
print(classification_report(y_test, y_hat_test))    
    
