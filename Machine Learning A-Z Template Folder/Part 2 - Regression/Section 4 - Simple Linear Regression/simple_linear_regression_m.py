# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('data_maior_teste.csv', sep=';')
X = dataset.iloc[:, :-1].values # Years of Experience
y = dataset.iloc[:, 1].values # Salary

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.0001, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the training Set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the results
y_pred = regressor.predict(X_test)

# Training Results
plt.scatter(X_train, y_train, color = 'black')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Road vs Air(Training)')
plt.xlabel('Road')
plt.ylabel('Air')
plt.show()

# Test Results
plt.scatter(X_test, y_test, color = 'black')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Salary vs Experience(Training)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#Scoring
print("\tPrecision: %1.3f" % r2_score(y_test, y_pred))