import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

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
print("====================random forest=====================")

rf_fit = RandomForestClassifier(n_estimators=5000, criterion='gini', max_depth=5,
                                min_samples_split=2, bootstrap=True, max_features='auto',
                                random_state=77, min_samples_leaf=1, class_weight={0:0.3, 1:0.7})
rf_fit.fit(x_train, y_train)
y_hat = rf_fit.predict(x_train)
y_hat_test = rf_fit.predict(x_test)
print(pd.crosstab(y_train, y_hat, rownames=['actual'], colnames=['predict']))
print(accuracy_score(y_train, y_hat))
print(classification_report(y_train, y_hat))
print()
print(pd.crosstab(y_test, y_hat_test, rownames=['actual'], colnames=['predict']))
print(accuracy_score(y_test, y_hat_test))
print(classification_report(y_test, y_hat_test))    
print("=========================graph=============================")

## 지니의 평균값 감소를 이용한 변수 중요도 그래프 ##
model_ranks = pd.Series(rf_fit.feature_importances_, index=x_train.columns, name='Importance').sort_values(ascending=False, inplace=False)
model_ranks.index.name = 'Variables'
top_features = model_ranks.iloc[:31].sort_values(ascending=True, inplace=False)
#plt.figure(figsize=(20,10))
#ax = top_features.plot(kind='barh')
#ax.set_title("Variable Importance plot")
#ax.set_xlabel("Mean decrease in Var")
#ax.set_yticklabels(top_features.index, fontsize=13)

''' # grid search는 pipeline # 3.9분 소요
### grid search ###
pipeline = Pipeline([('clf', RandomForestClassifier(criterion='gini', class_weight={0:0.3, 1:0.7}))])
parameters = {'clf__n_estimators':(2000,3000,5000),
              'clf__max_depth':(5,15,30),
              'clf__min_samples_split':(2,3),
              'clf__min_samples_leaf':(1,2)}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=5, verbose=1, scoring='accuracy')
grid_search.fit(x_train, y_train)
print('best score', grid_search.best_score_)
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))
predictions = grid_search.predict(x_test)
print()
print(accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(pd.crosstab(y_test, predictions, rownames=['actual'], colnames=['predict']))
print("=====================================Boosting================================")
'''
print("===============================AdaBoosting================================")

## adaBoost ##
dtree = DecisionTreeClassifier(criterion='gini', max_depth=1)
adabst_fit = AdaBoostClassifier(base_estimator=dtree, n_estimators=5000, learning_rate=0.05,
                                random_state=77)
adabst_fit.fit(x_train, y_train)
y_hat = adabst_fit.predict(x_train)
y_hat_test = adabst_fit.predict(x_test)
print(pd.crosstab(y_train, y_hat, rownames=['actual'], colnames=['predict']))
print(accuracy_score(y_train, y_hat))
print(classification_report(y_train, y_hat))
print()
print(pd.crosstab(y_test, y_hat_test, rownames=['actual'], colnames=['predict']))
print(accuracy_score(y_test, y_hat_test))
print(classification_report(y_test, y_hat_test))    
print("=============================GradientBoosting===============================")

## GradBoost ##
gbc_fit = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=5000,
                                     min_samples_split=2, min_samples_leaf=1, max_depth=1,
                                     random_state=77)
gbc_fit.fit(x_train, y_train)
y_hat = gbc_fit.predict(x_train)
y_hat_test = gbc_fit.predict(x_test)
print(pd.crosstab(y_train, y_hat, rownames=['actual'], colnames=['predict']))
print(accuracy_score(y_train, y_hat))
print(classification_report(y_train, y_hat))
print()
print(pd.crosstab(y_test, y_hat_test, rownames=['actual'], colnames=['predict']))
print(accuracy_score(y_test, y_hat_test))
print(classification_report(y_test, y_hat_test))    
print("=============================XGBoosting===============================")

xgb_fit = xgb.XGBClassifier(max_depth=2, n_estimators=5000, learning_rate=0.05)
xgb_fit.fit(x_train, y_train)
y_hat = xgb_fit.predict(x_train)
y_hat_test = xgb_fit.predict(x_test)
print(pd.crosstab(y_train, y_hat, rownames=['actual'], colnames=['predict']))
print(accuracy_score(y_train, y_hat))
print(classification_report(y_train, y_hat))
print()
print(pd.crosstab(y_test, y_hat_test, rownames=['actual'], colnames=['predict']))
print(accuracy_score(y_test, y_hat_test))
print(classification_report(y_test, y_hat_test))    
print("=============================Ensemble===============================")