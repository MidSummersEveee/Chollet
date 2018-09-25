from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# load source
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# shape info
# print(train_data.shape)
# >>> (60000, 28, 28)

# build network
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
model.add(layers.Dense(10, activation='softmax'))

# compile
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# reshape
train_data = train_data.reshape((60000, 28 * 28))
train_data = train_data.astype('float32') / 255

test_data = test_data.reshape((10000, 28 * 28))
test_data = test_data.astype('float32') / 255

# one-hot encoding 10 classes
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# learn
model.fit(
	train_data,
	train_labels,
	epochs=5,
	batch_size=128)

# evaluate
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'test_acc: {test_acc}')