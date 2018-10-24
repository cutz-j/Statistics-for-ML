### 최대마진분류기 ###
import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

os.chdir("d:/Github/Statistics-for-ML/Chapter06/")

## 데이터 전처리 ##
letter = pd.read_csv("letterdata.csv")
#columns = ['lettr', 'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar',
#           'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']
#letter.columns = columns

x_vars = letter.drop(['letter'], axis=1) # shape(20000, 16)
y_var = letter["letter"] # shape(20000, 1)

y_var = y_var.replace({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':10,
'K':11,'L':12,'M':13,'N':14,'O':15,'P':16,'Q':17,'R':18,'S':19,'T':20,
'U':21,'V':22,'W':23,'X':24,'Y':25,'Z':26})

# train/test split #
x_train, x_test, y_train, y_test = train_test_split(x_vars, y_var, train_size=0.7, random_state=77)

## 최대 마진 ##
#svm_fit = SVC(kernel='linear', C=1.0, random_state=77)
#svm_fit.fit(x_train, y_train)

# 87.4 % #
#print(pd.crosstab(y_train, svm_fit.predict(x_train), rownames=["Actual"], colnames=["Predicted"]))
#print("\nTrain accuracy: ", round(accuracy_score(y_train, svm_fit.predict(x_train)), 3))
#print("\n", classification_report(y_train, svm_fit.predict(x_train)))

# 85.6% #
#print(pd.crosstab(y_test, svm_fit.predict(x_test), rownames=["Actual"], colnames=["Predicted"]))
#print("\nTrain accuracy: ", round(accuracy_score(y_test, svm_fit.predict(x_test)), 3))
#print("\n", classification_report(y_test, svm_fit.predict(x_test)))

### 다항 커널 ###
#svm_poly_fit = SVC(kernel='poly', C=1.0, degree=2)
#svm_poly_fit.fit(x_train, y_train)

# 99% #
#print(pd.crosstab(y_train, svm_poly_fit.predict(x_train), rownames=["Actual"], colnames=["Predicted"]))
#print("\nTrain accuracy: ", round(accuracy_score(y_train, svm_poly_fit.predict(x_train)), 3))
#print("\n", classification_report(y_train, svm_poly_fit.predict(x_train)))

# 95% #
#print(pd.crosstab(y_test, svm_poly_fit.predict(x_test), rownames=["Actual"], colnames=["Predicted"]))
#print("\nTrain accuracy: ", round(accuracy_score(y_test, svm_poly_fit.predict(x_test)), 3))
#print("\n", classification_report(y_test, svm_poly_fit.predict(x_test)))
#
#svm_rbf_fit = SVC(kernel='rbf', C=1.0, gamma=0.1)
#svm_rbf_fit.fit(x_train, y_train)
#print(pd.crosstab(y_test, svm_rbf_fit.predict(x_test), rownames=["Actual"], colnames=["Predicted"]))
#print("\nTrain accuracy: ", round(accuracy_score(y_test, svm_rbf_fit.predict(x_test)), 3))
#print("\n", classification_report(y_test, svm_rbf_fit.predict(x_test)))

## 그리드 검색 ## --> parameters: C / gamma
pipeline = Pipeline([('clf', SVC(kernel='rbf', C=1, gamma=0.1))])
parameters = {'clf__C':(0.1, 0.3, 1, 3, 10, 30), 'clf__gamma': (0.001, 0.01, 0.1, 0.3, 1)}

grid_search_rbf = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=5, verbose=1, scoring='accuracy')
grid_search_rbf.fit(x_train, y_train)

print("score: %0.3f" % grid_search_rbf.best_score_)
best_parameters = grid_search_rbf.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
    predictions = grid_search_rbf.predict(x_test)
    print(round(accuracy_score(y_test, predictions), 4))
    print(classification_report(y_test, predictions))
    print(pd.crosstab(y_test, predictions, rownames=["Actual"], colnames=["Predicted"]))









