from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# load data from IMDB source
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
	num_words=10000)

# data is encoded as indices of words
# print(train_data[0])

# labels are binary repre pos(1) or neg(0)
# print(train_labels[13])


# review reconstruction
# (from int-seq to word-seq)

# reverse word-int map to int-word pattern
# word_index = imdb.get_word_index()
# reverse_word_index = dict(
# 	[(value, key) for (key, value) in word_index.items()])

# decoded review
# decoded_review = ' '.join(
# 	[reverse_word_index.get(i-3, '?') for i in train_data[0]])
# print(decoded_review)


# One-hot code all samples
# map seq [2,4] to [0,0,1,0,1,....,0]
# result has shape(samples, dimensions)

# self-version longer but more clear
# def vectorize_sequences(sequences, dimension=10000):
# 	results = np.zeros((len(sequences), dimension))
# 	for i, sequence in enumerate(sequences):
# 		for index in sequence:
# 			results[i, index] = 1.
# 	return results

# alternate implementation
def vectorize_sequences(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results

# vectorization
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# print(x_train[0])

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

# define validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# tranning
history = model.fit(
	partial_x_train,
	partial_y_train,
	epochs=20,
	batch_size=512,
	validation_data=(x_val, y_val))

# monitor results
history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# plotting loss trend
plt.plot(epochs, loss_values, 'bo', label='Tranning loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
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