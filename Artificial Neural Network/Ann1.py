# Preproceesing the Data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Handling the categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
LabelEncoder_x_1 = LabelEncoder()
x[:, 1] = LabelEncoder_x_1.fit_transform(x[:, 1])
LabelEncoder_x_2 = LabelEncoder()
x[:, 2] = LabelEncoder_x_2.fit_transform(x[:, 2])


# Splitting the dataset into training and testing datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0 )

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

# Preparing the ANN and Adding the input layer(First hidden layer)
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Third hidden layer(output layer)
classifier.add(Dense(optimizer = 'adam', loss = 'binary_crossentropy', metrice = ['accuracy']))

# Fitting the ANN
classifier.fit(x_train, y_train, batch_size, nb_epoch = 100)

# Prediction
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)