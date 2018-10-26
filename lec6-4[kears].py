import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
### 케라스 ###
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta, Adam, RMSprop
from keras.utils import np_utils

np.random.seed(777)

## 데이터 전처리 ##
digits = load_digits()
X = digits.data # shape(1797, 64)
y = digits.target # shape(1797,)

## 이미지 출력 ##
#plt.matshow(digits.images[0])
#plt.show()

## 표준화 ##
x_scale = StandardScaler().fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, train_size=0.7, random_state=77)

# hyper parameter #
classes = 10
batch_size = 30
epochs = 500

# n 차원 벡터 생성 -> value = 10 # --> one-hot
Y_train = np_utils.to_categorical(y_train, classes)

## 딥러닝 모델 구축 ##
model = Sequential()

## 신경망 구축 ##
# 입력 노드 #
model.add(Dense(100, input_shape=(64,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 은닉층 #
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 출력 레이어 #
model.add(Dense(classes))
model.add(Activation('softmax'))

## cost 함수 & gradient descent ##
model.compile(loss='categorical_crossentropy', optimizer='sgd')

# training # mini-batch: 120/1257 // epochs: 200
model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1)

## 모델 예측 ##
y_train_predclass = model.predict_classes(x_train, batch_size=batch_size)
y_test_predclass = model.predict_classes(x_test, batch_size=batch_size)
print("train acc: ", round(accuracy_score(y_train, y_train_predclass), 4))
print("train class rep\n", classification_report(y_train, y_train_predclass))
print("\nconfusion matrix\n", pd.crosstab(y_train, y_train_predclass, rownames=["actual"],
                                          colnames=["predicted"]))

print("test acc: ", round(accuracy_score(y_test, y_test_predclass), 4))
print("test class rep\n", classification_report(y_test, y_test_predclass))
print("\nconfusion matrix\n", pd.crosstab(y_test, y_test_predclass, rownames=["actual"],
                                          colnames=["predicted"]))

















