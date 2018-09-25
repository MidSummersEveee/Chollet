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


# K-fold cross validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []


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
	history = model.fit(
		partial_train_data,
		partial_train_labels,
		validation_data=(val_data, val_labels),
		epochs=num_epochs,
		batch_size=1,
		verbose=0)
	mae_history = history.history['val_mean_absolute_error']
	all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# plotting
# the following is difficult to see!
# so we go for a exponetial moving average of previous points
# plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('validation MAE')
# plt.show()

# print(f'length of all_mae_histories is {len(all_mae_histories)}')
# len would be 4

def smooth_curve(points, factor=0.9):
	smoothed_points =[]
	for point in points:
		if smoothed_points:
			previous = smoothed_points[-1]
			smoothed_points.append(previous * factor + point * (1 - factor))
		else:
			smoothed_points.append(point)
	return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('validation MAE')