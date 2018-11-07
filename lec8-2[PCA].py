import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

digits = load_digits()
X = digits.data # --> 8x8 픽셀
y = digits.target

#plt.matshow(digits.images[0])
#plt.show()

## scale ##
X_scale = scale(X, axis=0) # scaling

pca = PCA(n_components=2)
reduced_x = pca.fit_transform(X_scale)