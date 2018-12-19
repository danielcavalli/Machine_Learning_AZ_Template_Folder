# D A T A  P R E P R O C E S S I N G
# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.preprocessing import StandardScaler # Featurizing
from sklearn.model_selection import train_test_split # Training

# Getting the DataSet
dataSet = pd.read_csv('Data.csv')
x = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, 3].values

# Splitting the Dataset: Train and Test data ramification
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Standardisation of data: x = (x - mean(x))/standard_deviation(x)
# Normalization of data: x = (x - min(x))/(max(x) - min(x))
"""sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)"""