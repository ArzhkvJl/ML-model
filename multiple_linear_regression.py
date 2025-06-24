import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def rmse(y_true, y_pred):
    error = (y_true - y_pred) ** 2
    return np.sqrt(np.mean(error))


X = np.array(
    [[5, 100], [4.9, 90], [3.8, 20], [4.2, 70], [4.8, 85], [3.7, 68], [4.6, 82], [4.3, 86], [3.9, 99], [4.1, 82],
     [4.7, 71], [3.2, 32], [3.5, 45], [3.3, 58], [4.5, 77]])
Y = np.array([1, 20, 6, 5, 4, 18, 2, 9, 10, 5, 3, 13, 12, 2, 14])

fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], Y, color='firebrick')

model = LinearRegression().fit(X, Y)
r_sq = model.score(X, Y)
print('coefficient of determination:', r_sq)
Yhat = model.predict(X)
print(Yhat)
y_pred = model.intercept_ + np.sum(model.coef_ * X, axis=1)
errors = np.zeros((15, 1))
for i in range(15):
    errors[i][0] = (rmse(Y[i], y_pred[i]))

while r_sq <= 0.5:
    error_ind = (np.argmax(errors))
    errors = np.delete(errors, error_ind)
    X = np.delete(X, error_ind)
    X = np.delete(X, error_ind).reshape(-1, 2)
    Y = np.delete(Y, error_ind)
    model = LinearRegression().fit(X, Y)
    r_sq = model.score(X, Y)

y_pred = model.predict(X)
print('coefficient of determination:', r_sq)
print(X)
print(y_pred)

ax.scatter(X[:, 0].flatten(), X[:, 1].flatten(), y_pred, facecolor=(0, 0, 0, 0), s=30, edgecolor='b')
coef_X1 = model.coef_[0]
coef_X2 = model.coef_[1]
x1_surf, x2_surf = np.meshgrid(np.linspace(X[:,0].min(), X[:,0].max(), 50),
                                np.linspace(X[:,1].min(), X[:,1].max(), 50))
y_surf = model.intercept_ + coef_X1 * x1_surf + coef_X2 * x2_surf
ax.plot_surface(x1_surf, x2_surf, y_surf, color='cyan', alpha=0.5, label='Regression Plane')

plt.show()
