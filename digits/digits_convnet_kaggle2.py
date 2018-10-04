from keras.models import Sequential
from keras.utils import np_utils
from keras import models
from keras import layers
from keras import optimizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data
train = pd.read_csv('../../source/train.csv')
labels = train.ix[:,0].values.astype('int32')
x_train = (train.ix[:,1:].values).astype('float32')
x_test = (pd.read_csv('../../source/test.csv').values).astype('float32')


# one hot
y_train = np_utils.to_categorical(labels)


# pre-processing: divide by max and substract mean
scale = np.max(x_train)
x_train /= scale
x_test /= scale

mean = np.std(x_train)
x_train -= mean
x_test -= mean


# build network
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# compile
model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])


# reshape
x_train = x_train.reshape((42000, 28, 28, 1))
x_train = x_train.astype('float32')

x_test = x_test.reshape((28000, 28 ,28, 1))
x_test = x_test.astype('float32')



print("Learning...")
# history = model.fit(x_train, y_train, epochs=10, batch_size=16)
history = model.fit(x_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

acc  = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Tranning Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training & Validation Loss')
plt.legend()

plt.show()

model.save('kaggle_aug_drop_1.h5')


# print("Generating test predictions...")
# preds = model.predict_classes(x_test, verbose=0)

# def write_preds(preds, fname):
#     pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

# write_preds(preds, "predc-keras-convnet3.csv")