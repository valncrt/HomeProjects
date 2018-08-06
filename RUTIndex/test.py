import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense

from keras.preprocessing import sequence
from keras.utils import np_utils

base_series = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
series = base_series * 10
seq_length = len(base_series)
X = []
Y = []


def unit(index): return [1.0 if i == index else 0.0 for i in range(seq_length)]


# make buckets
for i in range(0, len(series) - seq_length, 1):
    X.append(series[i:i + seq_length])
    Y.append(unit(np.mod(i, seq_length)))
X = np.array(X)
Y = np.array(Y)

model = Sequential()

model.add(Dense(seq_length, input_dim=X.shape[1], init='normal', activation='softmax'))
# try alternatives if you wish
# model.add(Dense(30,input_dim=X.shape[1], activation="relu", init='normal'))
# model.add(Dense(seq_length, init='normal', activation='softmax'))

model.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X, Y, nb_epoch=350, verbose=0)
scores = model.evaluate(X, Y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1] * 100))