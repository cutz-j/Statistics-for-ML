### ANN ###
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, w):
    return 1 / (1 + np.exp(-x * w))

def tanh(x, w):
    return 2 * sigmoid(2*x, w) - 1

x = np.arange(-5, 5, 0.1)
w = np.random.normal(1)

tanh_val = tanh(x, w)

plt.figure()
plt.plot(x, tanh_val)