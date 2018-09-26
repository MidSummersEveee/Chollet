from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

# load source
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# data shape
# img_height, img_width, img_channels


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

model.summary()



# reshape
train_data = train_data.reshape((60000, 28, 28, 1))
train_data = train_data.astype('float32') / 255

test_data = test_data.reshape((10000, 28 ,28, 1))
test_data = test_data.astype('float32') / 255

# one-hot encoding 10 classes
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# compile
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_data, test_labels)

print(test_acc)