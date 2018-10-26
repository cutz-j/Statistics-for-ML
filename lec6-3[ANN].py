import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

### 숫자 분류 ANN ### -> sklearn
## 데이터 전처리 ##
digits = load_digits()
X = digits.data # shape (1797, 64)
y = digits.target # shape (1797, )

## 숫자 그래프 ##
#plt.matshow(digits.images[0])
#plt.show()

x_vars_stdscle = StandardScaler().fit_transform(X) # 표준화 --> fit + transform
# split #
x_train, x_test, y_train, y_test = train_test_split(x_vars_stdscle, y, train_size=0.7, random_state=77)

# pipe line #
pipeline = Pipeline([('mlp', MLPClassifier(hidden_layer_sizes=(100,50,), activation='relu',
                                          solver='adam', alpha=0.0001, max_iter=300))])

parameters = {'mlp__activation':('relu', 'logistic'),
              'mlp__solver':('adam', 'sgd'),
              'mlp__alpha': (0.001, 0.01, 0.1, 0.3, 0.5, 1.0),
              'mlp__max_iter': (100, 200, 300)}

# 그리드 실행 #
grid_search_nn = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=5, verbose=1, scoring='accuracy')
grid_search_nn.fit(x_train, y_train)

# best acc score #
print(grid_search_nn.best_score_)

# parameter 비교
best_parameters = grid_search_nn.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))

predictions_train = grid_search_nn.predict(x_train)
predictions_test = grid_search_nn.predict(x_test)

# train set score & confusion matrix #
print("acc: ", accuracy_score(y_train, predictions_train))
print("\n\nNN\n", pd.crosstab(y_train, predictions_train, rownames=["actual"], colnames=["predicted"]))

# test set score % confusion matrix #
print("\nacc: ", accuracy_score(y_test, predictions_test))
print("\n\nNN\n", pd.crosstab(y_test, predictions_test, rownames=["actual"], colnames=["predicted"]))
