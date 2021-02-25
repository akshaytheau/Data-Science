# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

data=pd.read_csv('diabetes.csv')
data.head()


X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(16,activation='relu',input_dim=8))
classifier.add(Dropout(0.2))

#adding the second hidden layer
classifier.add(Dense(16,activation='relu'))
classifier.add(Dropout(0.2))

#adding the output layer
classifier.add(Dense(1,activation='sigmoid'))

classifier.compile(optimizer="Adam", loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=100, epochs=300)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#sns.heatmap(cm,annot=True)
#plt.savefig('h.png')
