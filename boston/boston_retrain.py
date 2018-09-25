from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# load source
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()


# Normalization
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


# Modeling
def build_model():
	model = models.Sequential()
	model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(1))
	model.compile(
		optimizer='rmsprop',
		loss='mse',
		metrics=['mae'])
	return model


# Tranning
model = build_model()
model.fit(
	train_data,
	train_labels,
	epochs=80,
	batch_size=16,
	verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_data, test_labels)

# output
print(test_mae_score)