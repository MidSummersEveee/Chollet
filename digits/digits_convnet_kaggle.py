from keras.models import Sequential
from keras.utils import np_utils
from keras import models
from keras import layers
import pandas as pd
import numpy as np

# Read data
train = pd.read_csv('../../source/train.csv')
labels = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (pd.read_csv('../../source/test.csv').values).astype('float32')

# convert list of labels to binary class matrix
y_train = np_utils.to_categorical(labels) 

# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.std(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

# build network
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# compile
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# reshape
X_train = X_train.reshape((42000, 28, 28, 1))
X_train = X_train.astype('float32')

X_test = X_test.reshape((28000, 28 ,28, 1))
X_test = X_test.astype('float32')


print("Learning...")
model.fit(X_train, y_train, epochs=5, batch_size=16, validation_split=0.1)




print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=0)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "predc-keras-convnet.csv")