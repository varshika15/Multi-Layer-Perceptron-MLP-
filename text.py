# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:49:34 2018

@author: varshika
"""


from keras.models import Sequential
from keras.layers import Dense
import numpy

numpy.random.seed(7)
# load data set
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# training/test data split
X = dataset[:,0:8]
Y = dataset[:,8]
# model building
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)
# evaluating model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
