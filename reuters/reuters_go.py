from keras.datasets import reuters
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# load source
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# vecotrization
def vectorize_sequences(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# one-hot encoding labels
def to_one_hot(labels, dimension=46):
	results = np.zeros((len(labels), dimension))
	for i, label in enumerate(labels):
		results[i, label] = 1
	return results

one_hot_train_lables = to_one_hot(train_labels)
one_hot_test_lables = to_one_hot(test_labels)

# modeling
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# compiling
model.compile(optimizer='rmsprop',
	loss='categorical_crossentropy',
	metrics=['accuracy'])


# define validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_lables[:1000]
partial_y_train = one_hot_train_lables[1000:]

# tranning
history = model.fit(
	partial_x_train,
	partial_y_train,
	epochs=20,
	batch_size=512,
	validation_data=(x_val, y_val))


# monitor results
history_dict = history.history

loss = history_dict['loss']
val_loss = history_dict['val_loss']

acc = history_dict['acc']
val_acc = history_dict['val_acc']

epochs = range(1, len(loss) + 1)

# plotting loss trend
plt.plot(epochs, loss, 'bo', label='Tranning loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Tranning & validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# plotting acc trend
plt.clf()
plt.plot(epochs, acc, 'bo', label='Tranning acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Tranning & validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()