from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np

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


# K-fold cross validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []


# import warnings
# # put the following 2 lines before the nested loop code
# with warnings.catch_warnings():
# 	warnings.simplefilter("ignore", category=RuntimeWarning)



for i in range(k):
	print(f"processing fold {i}.....")
	val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
	val_labels = train_labels[i * num_val_samples: (i + 1) * num_val_samples]

	partial_train_data = np.concatenate(
		[train_data[:i * num_val_samples],
		train_data[(i + 1) * num_val_samples:]],
		axis=0)
	partial_train_labels = np.concatenate(
		[train_labels[:i * num_val_samples],
		train_labels[(i + 1) * num_val_samples:]],
		axis=0)

	model = build_model()
	model.fit(
		partial_train_data,
		partial_train_labels,
		epochs=num_epochs,
		batch_size=1,
		verbose=0)
	val_mse, val_mae = model.evaluate(val_data, val_labels, verbose=0)
	all_scores.append(val_mae)

print()
print(all_scores)
print(np.mean(all_scores))