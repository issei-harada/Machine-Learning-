# Polynomial Regression

# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
#no need here
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)"""
# Feature Scaling

# Fitting Linear Regression to the dataset as a comparison

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting Polynomial Regression to the dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y) 

plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression')


plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'green')
plt.title('Polynomial Regression')
"""
#Better visualisation
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X,y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'yellow')
plt.title('Polynomial Regression')
plt.show()"""

y_pred = lin_reg.predict(X)
y_pred_2 = lin_reg_2.predict(poly_reg.fit_transform(X))

plt.scatter(X,y, color = 'red')
plt.scatter(X,y_pred, color = 'blue')
plt.scatter(X,y_pred_2, color = 'cyan')

test = [4,5,6.6,7]
lin_reg.predict(test)