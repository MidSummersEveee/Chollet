from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# load data from IMDB source
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
	num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results

# vectorization
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# modeling
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# compiling
model.compile(optimizer='rmsprop',
	loss='binary_crossentropy',
	metrics=['acc'])

# tranning
history = model.fit(
	x_train,
	y_train,
	epochs=4,
	batch_size=512,
	validation_data=(x_test, y_test))

# eva
results = model.evaluate(x_test, y_test)
print(results)