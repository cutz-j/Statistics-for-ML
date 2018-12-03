### deep autoencoder -> mnist ###
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from keras.layers import Input, Dense
from keras.models import Model

## data load ##
digits = load_digits()
X = digits.data # shape(1797, 64)
y = digits.target # shape(1797, 1)

X_scale = StandardScaler().fit_transform(X)

## 신경망 구성 ##
input_layer = Input(shape=(64,), name="input")
encoded = Dense(32, activation='relu', name='h1encode')(input_layer)
encoded = Dense(16, activation='relu', name='h2encode')(encoded)
encoded = Dense(8, activation='relu', name='h3encode')(encoded)
encoded = Dense(2, activation='relu', name='h4latent_layer')(encoded)
decoded = Dense(8, activation='relu', name='h3decode')(encoded)
decoded = Dense(16, activation='relu', name='h4decode')(decoded)
decoded = Dense(32, activation='relu', name='h5decode')(decoded)
decoded = Dense(64, activation='sigmoid', name='h6decode')(decoded)

## 학습 ##
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scale, X_scale, epochs=100, batch_size=256, shuffle=True, validation_split=0.2)

## autoencoder 생성 ##
encoder = Model(autoencoder.input, autoencoder.get_layer("h4latent_layer").output)
reduced_X = encoder.predict(X_scale)

## 시각화 ##
zero_x, zero_y = [],[] ; one_x, one_y = [],[]
two_x,two_y = [],[]; three_x, three_y = [],[]
four_x,four_y = [],[]; five_x,five_y = [],[]
six_x,six_y = [],[]; seven_x,seven_y = [],[]
eight_x,eight_y = [],[]; nine_x,nine_y = [],[]


for i in range(len(reduced_X)):
    if y[i] == 0:
        zero_x.append(reduced_X[i][0])
        zero_y.append(reduced_X[i][1])
        
    elif y[i] == 1:
        one_x.append(reduced_X[i][0])
        one_y.append(reduced_X[i][1])

    elif y[i] == 2:
        two_x.append(reduced_X[i][0])
        two_y.append(reduced_X[i][1])

    elif y[i] == 3:
        three_x.append(reduced_X[i][0])
        three_y.append(reduced_X[i][1])

    elif y[i] == 4:
        four_x.append(reduced_X[i][0])
        four_y.append(reduced_X[i][1])

    elif y[i] == 5:
        five_x.append(reduced_X[i][0])
        five_y.append(reduced_X[i][1])

    elif y[i] == 6:
        six_x.append(reduced_X[i][0])
        six_y.append(reduced_X[i][1])

    elif y[i] == 7:
        seven_x.append(reduced_X[i][0])
        seven_y.append(reduced_X[i][1])

    elif y[i] == 8:
        eight_x.append(reduced_X[i][0])
        eight_y.append(reduced_X[i][1])
    
    elif y[i] == 9:
        nine_x.append(reduced_X[i][0])
        nine_y.append(reduced_X[i][1])



zero = plt.scatter(zero_x, zero_y, c='r', marker='x',label='zero')
one = plt.scatter(one_x, one_y, c='g', marker='+')
two = plt.scatter(two_x, two_y, c='b', marker='s')

three = plt.scatter(three_x, three_y, c='m', marker='*')
four = plt.scatter(four_x, four_y, c='c', marker='h')
five = plt.scatter(five_x, five_y, c='r', marker='D')

six = plt.scatter(six_x, six_y, c='y', marker='8')
seven = plt.scatter(seven_x, seven_y, c='k', marker='*')
eight = plt.scatter(eight_x, eight_y, c='r', marker='x')

nine = plt.scatter(nine_x, nine_y, c='b', marker='D')


plt.legend((zero,one,two,three,four,five,six,seven,eight,nine),
           ('zero','one','two','three','four','five','six','seven','eight','nine'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=10)

plt.xlabel('PC 1')
plt.ylabel('PC 2')

plt.show()


## 3차원 잠재 변수 ##
input_layer = Input(shape=(64,), name="input")
encoded = Dense(32, activation='relu', name='h1encode')(input_layer)
encoded = Dense(16, activation='relu', name='h2encode')(encoded)
encoded = Dense(8, activation='relu', name='h3encode')(encoded)
encoded = Dense(3, activation='relu', name='h4latent_layer')(encoded)
decoded = Dense(8, activation='relu', name='h3decode')(encoded)
decoded = Dense(16, activation='relu', name='h4decode')(decoded)
decoded = Dense(32, activation='relu', name='h5decode')(decoded)
decoded = Dense(64, activation='sigmoid', name='h6decode')(decoded)

## 학습 ##
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scale, X_scale, epochs=100, batch_size=256, shuffle=True, validation_split=0.2)

## autoencoder 생성 ##
encoder = Model(autoencoder.input, autoencoder.get_layer("h4latent_layer").output)
reduced_X3D = encoder.predict(X_scale)

zero_x, zero_y,zero_z = [],[],[] ; one_x, one_y,one_z = [],[],[]
two_x,two_y,two_z = [],[],[]; three_x, three_y,three_z = [],[],[]
four_x,four_y,four_z = [],[],[]; five_x,five_y,five_z = [],[],[]
six_x,six_y,six_z = [],[],[]; seven_x,seven_y,seven_z = [],[],[]
eight_x,eight_y,eight_z = [],[],[]; nine_x,nine_y,nine_z = [],[],[]



for i in range(len(reduced_X3D)):
    
    if y[i]==10:
        continue
    
    elif y[i] == 0:
        zero_x.append(reduced_X3D[i][0])
        zero_y.append(reduced_X3D[i][1])
        zero_z.append(reduced_X3D[i][2])
        
    elif y[i] == 1:
        one_x.append(reduced_X3D[i][0])
        one_y.append(reduced_X3D[i][1])
        one_z.append(reduced_X3D[i][2])

    elif y[i] == 2:
        two_x.append(reduced_X3D[i][0])
        two_y.append(reduced_X3D[i][1])
        two_z.append(reduced_X3D[i][2])

    elif y[i] == 3:
        three_x.append(reduced_X3D[i][0])
        three_y.append(reduced_X3D[i][1])
        three_z.append(reduced_X3D[i][2])

    elif y[i] == 4:
        four_x.append(reduced_X3D[i][0])
        four_y.append(reduced_X3D[i][1])
        four_z.append(reduced_X3D[i][2])

    elif y[i] == 5:
        five_x.append(reduced_X3D[i][0])
        five_y.append(reduced_X3D[i][1])
        five_z.append(reduced_X3D[i][2])

    elif y[i] == 6:
        six_x.append(reduced_X3D[i][0])
        six_y.append(reduced_X3D[i][1])
        six_z.append(reduced_X3D[i][2])

    elif y[i] == 7:
        seven_x.append(reduced_X3D[i][0])
        seven_y.append(reduced_X3D[i][1])
        seven_z.append(reduced_X3D[i][2])

    elif y[i] == 8:
        eight_x.append(reduced_X3D[i][0])
        eight_y.append(reduced_X3D[i][1])
        eight_z.append(reduced_X3D[i][2])
    
    elif y[i] == 9:
        nine_x.append(reduced_X3D[i][0])
        nine_y.append(reduced_X3D[i][1])
        nine_z.append(reduced_X3D[i][2])



# 3- Dimensional plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(zero_x, zero_y,zero_z, c='r', marker='x',label='zero')
ax.scatter(one_x, one_y,one_z, c='g', marker='+',label='one')
ax.scatter(two_x, two_y,two_z, c='b', marker='s',label='two')

ax.scatter(three_x, three_y,three_z, c='m', marker='*',label='three')
ax.scatter(four_x, four_y,four_z, c='c', marker='h',label='four')
ax.scatter(five_x, five_y,five_z, c='r', marker='D',label='five')

ax.scatter(six_x, six_y,six_z, c='y', marker='8',label='six')
ax.scatter(seven_x, seven_y,seven_z, c='k', marker='*',label='seven')
ax.scatter(eight_x, eight_y,eight_z, c='r', marker='x',label='eight')

ax.scatter(nine_x, nine_y,nine_z, c='b', marker='D',label='nine')

ax.set_xlabel('Latent Feature 1',fontsize = 13)
ax.set_ylabel('Latent Feature 2',fontsize = 13)
ax.set_zlabel('Latent Feature 3',fontsize = 13)

ax.set_xlim3d(0,60)

plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=10, bbox_to_anchor=(0, 0))

plt.show()



ax.set_xlim3d(left = 0,right = 30)
ax.set_ylim3d(left = 0,right = 30)
ax.set_zlim3d(left = 0,right = 30)















