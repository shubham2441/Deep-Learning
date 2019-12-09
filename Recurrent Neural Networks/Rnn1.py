# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:59:11 2019

@author: shubh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the Dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Preprocess the Data, bringing all the values within the range of 0 and 1(Normalizint hte Data)
from sklearn.preprocessing import MinMaxScaler

#MinMaxScaler is the in-built function which helps to bring the data in the defined range
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a dataset for 60 timesteps and 1 output(the previous 60 entries will help the machine to predict the next output)
x_train = []
y_train = []
for i in range(60, training_set_scaled.shape[0]):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
# Adding a LSTM Layer (unit = no. of neuron)
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
#Dropout regularization = some neurons will be dropeed out during an iteration to avoid overfitting
regressor.add(Dropout(0.2))

#Second LSTM Layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Third LSTM Layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Fourth LSTM Layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Output Layer
regressor.add(Dense(units = 1))

# Compiling with the help of the optimizer
# Optimizer = adam(it the name of the stochastic gradient descent optimizer)
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit the model in the LSTM
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)

#Predicting the Google Stock Price for 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_train.iloc[:, 1:2].values




