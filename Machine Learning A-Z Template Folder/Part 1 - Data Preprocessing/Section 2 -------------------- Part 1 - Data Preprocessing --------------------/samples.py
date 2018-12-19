# Encoding Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_x = LabelEncoder()
x[:, 0] = label_x.fit_transform(x[:, 0])
onehot = OneHotEncoder(categorical_features = [0])
x = onehot.fit_transform(x).toarray()
label_y = LabelEncoder()
y = label_y.fit_transform(y)

# Cleaning Data
from sklearn.preprocessing import Imputer as imp #Cleaning
imputer = imp(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

