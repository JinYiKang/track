import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import QuantileRegressor

data = np.loadtxt('./2.txt', dtype=float)
X = np.array(data[:,0])[100:300]
X = np.subtract(X, X[0])
Y = np.array(data[:, 13])[100:300]

X_train = [[x, x**2, x**3, x**4, x**5, x**6] for x in X]
X_train = np.array(X_train)
Y_train = Y
reg = QuantileRegressor(quantile=0.5, solver='highs').fit(X_train, Y_train)

Y_pred = reg.predict(X_train)


plt.plot(X,Y)
plt.plot(X, Y_pred)
plt.show()