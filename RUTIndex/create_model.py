#import keras
import pandas as pd
import math as math
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import keras
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation,BatchNormalization
import numpy as np
import pandas as pd
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import keras
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier


weeks_in_look_back=12 #number of weeks to included to make prediction minus one
weeks_back=weeks_in_look_back
file ="C:\\Users\\csvai\\OneDrive\\Desktop\\Stephen\\RUTIndex\\data\\all_indexes_weekly_30_yrs_adj.csv"

pd=pd.read_csv(file)

print(pd.size)
size_for_even_batch_division=math.floor(pd.size/weeks_back)*weeks_back -weeks_back #don't run over the end
print("math.floor(pd.size/batch_size)*batch_size",math.floor(pd.size/weeks_back)*weeks_back)
print("(pd.size%batch_size)*batch_size",(pd.size%weeks_back)*weeks_back)

print("size_for_even_batch_division",size_for_even_batch_division)

dataset=[]
for i in range (0, size_for_even_batch_division, 1):
    line=list(pd.High[i:i+weeks_back])
    #print("Line= ",i,line)
    dataset.append(line)
    #print("I= ",i," \n",pd.High[i:i+batch_size])

#print(dataset)
numpy_array = np.asarray(dataset)
#numpy_array=dataset
#print ("numpy_array shape" ,numpy_array.shape,numpy_array)

X = numpy_array[:,0:weeks_back-1]
Y = numpy_array[:,weeks_back-1]


scaler = Normalizer().fit(X)
X = scaler.transform(X)

#scaler = StandardScaler().fit(X)
#X = scaler.transform(X)

#scaler = MinMaxScaler(feature_range=(0, 1))
#X = scaler.fit_transform(X)
#Y = scaler.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=np.random.seed(70))



print ("X shape",X_train.shape)
print(X[0:2])
print(Y[0:2])


layer_size = 500
dropout = .25
#learning_rate=.000425
learning_rate = .001
batch_size = 10
patience_int = 10
epochs_int = 250

def get_model():
    print('Build model...')
    layer_init='uniform'

    model = Sequential()
    #act = keras.layers.advanced_activations.PReLU(weights=None, alpha_initializer="zero")
    act=Activation('relu')
    model.add(Dense(layer_size, input_dim=weeks_back-1, kernel_initializer=layer_init))
    model.add(Dropout(dropout))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1))

    model.add(Dense(layer_size, input_dim=weeks_back-1, kernel_initializer=layer_init))
    model.add(Dropout(dropout))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1))

    model.add(Dense(layer_size, input_dim=weeks_back-1, kernel_initializer=layer_init))
    model.add(Dropout(dropout))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1))

    model.add(Dense(layer_size, input_dim=weeks_back-1, kernel_initializer=layer_init))
    model.add(Dropout(dropout))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    optim =keras.optimizers.Adam(lr=learning_rate)
    #optim=
    model.compile(loss='mean_squared_logarithmic_error', optimizer=optim, metrics=['accuracy'])
    #early_stopping = EarlyStopping(monitor='val_loss', patience=patience_int)


    #model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs_int, callbacks=[early_stopping])
    #score, acc = model.evaluate(X_test, y_test)
    #scores = model.evaluate(X[test], Y[test], verbose=0)
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    #print('Test score:', score)
    #print('Test accuracy:', acc)
    return model


#model=KerasRegressor(build_fn=build_model)
model=get_model()

early_stopping = EarlyStopping(monitor='val_loss', patience=20)
model.fit(X_train, Y_train, batch_size=batch_size, epochs=200, callbacks=[early_stopping],
          validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test)
print('Test score:', score)
#print('Test accuracy:', acc)

########
