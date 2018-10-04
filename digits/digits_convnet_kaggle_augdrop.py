from keras.preprocessing.image import ImageDataGenerator
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



# data augmentation
datagen = ImageDataGenerator(
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest')

train_generator = datagen.flow(x_train, y_train, batch_size=20)


print("Learning...")
history = model.fit_generator(
	train_generator,
	steps_per_epoch=len(x_train) / 20,
	epochs=30)


print("Generating test predictions...")
preds = model.predict_classes(x_test, verbose=0)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "predc-keras-convnet2.csv")