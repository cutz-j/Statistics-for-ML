import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

b_cancer = pd.read_csv("d:/data/ML/breast-cancer-wisconsin.data.txt", sep=',', header=None)
colList = ["ID number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
           "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin",
           "Normal Nucleoli", "Mitoses", "Class"]
for col in range(len(colList)):
    colList[col] = colList[col].replace(" ", "_")
b_cancer.columns = colList

# 결측치 대체 # --> ?를 nan으로, nan을 최빈값으로 #
b_cancer['Bare_Nuclei'] = b_cancer['Bare_Nuclei'].replace('?', np.NAN)
b_cancer['Bare_Nuclei'] = b_cancer['Bare_Nuclei'].fillna(b_cancer['Bare_Nuclei'].value_counts().index[0])
b_cancer['Cancer_Ind'] = 0
b_cancer.loc[b_cancer['Class'] == 4, 'Cancer_Ind'] = 1

# 무의미 변수 제거 #
x_vars = b_cancer.drop(['ID_number', 'Class', 'Cancer_Ind'], axis=1)
y_var = b_cancer['Cancer_Ind']
x_vars_stdscle = StandardScaler().fit_transform(x_vars.values) # 표준화( x - mean / std ) --> scaling
x_vars_stdscle_df = pd.DataFrame(x_vars_stdscle, index=x_vars.index, columns=x_vars.columns)
x_train, x_test, y_train, y_test = train_test_split(x_vars_stdscle_df, y_var, train_size=0.7, random_state=77)

## KNN ##
knn_fit = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')
knn_fit.fit(x_train, y_train)

#print("\n\nConfusion Matrix\n\n", pd.crosstab(y_train, knn_fit.predict(x_train), rownames=["Actuall"], colnames=["Predicted"]))
#print("\naccruacy: ", round(accuracy_score(y_train, knn_fit.predict(x_train)), 3)) # 학습데이터 98%
#print("\nreport: \n", classification_report(y_train, knn_fit.predict(x_train)))

print("\n\nConfusion Matrix\n\n", pd.crosstab(y_test, knn_fit.predict(x_test), rownames=["Actuall"], colnames=["Predicted"]))
print("\naccruacy: ", round(accuracy_score(y_test, knn_fit.predict(x_test)), 3)) # 학습데이터 97.6%
print("\nreport: \n", classification_report(y_test, knn_fit.predict(x_test)))

## KNN K tuning ##
k_vals = [i for i in range(1, 51, 2)]
dummyarray = np.empty((len(k_vals),3))
k_valchart = pd.DataFrame(dummyarray)
k_valchart.columns = ["K_value", "Train_acc", "Test_acc"]

for i in range(len(k_vals)):
    knn_fit = KNeighborsClassifier(n_neighbors=k_vals[i], p=2, metric='minkowski')
    knn_fit.fit(x_train, y_train)
    
    print("\n", k_vals[i])
    print("\n\nConfusion Matrix\n\n", pd.crosstab(y_train, knn_fit.predict(x_train), rownames=["Actuall"], colnames=["Predicted"]))
    print("\naccruacy: ", round(accuracy_score(y_train, knn_fit.predict(x_train)), 3)) # 학습데이터 97.6%
    print("\nreport: \n", classification_report(y_train, knn_fit.predict(x_train)))
    
    print("\n\nConfusion Matrix\n\n", pd.crosstab(y_test, knn_fit.predict(x_test), rownames=["Actuall"], colnames=["Predicted"]))
    print("\naccruacy: ", round(accuracy_score(y_test, knn_fit.predict(x_test)), 3)) # 학습데이터 97.6%
    print("\nreport: \n", classification_report(y_test, knn_fit.predict(x_test)))
    k_valchart.loc[i, 'K_value'] = k_vals[i]
    k_valchart.loc[i, 'Train_acc'] = round(accuracy_score(y_train, knn_fit.predict(x_train)), 3)
    k_valchart.loc[i, 'Test_acc'] = round(accuracy_score(y_test, knn_fit.predict(x_test)), 3)
    
# graph ##
plt.figure()
plt.plot(k_valchart["K_value"], k_valchart["Train_acc"], "r-")
plt.plot(k_valchart["K_value"], k_valchart["Test_acc"], "b-")

plt.axis([0.8, 5, 0.82, 1.005])

for a, b in zip(k_valchart["K_value"], k_valchart["Train_acc"]):
    plt.text(a,b,str(b), fontsize=10)
for a, b in zip(k_valchart["K_value"], k_valchart["Test_acc"]):
    plt.text(a,b,str(b), fontsize=10)
plt.legend(loc='upper right')
plt.show()
























