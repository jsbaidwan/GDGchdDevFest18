# Import the tensorFlow library
import tensorflow as tf


# Load and prepare MNIST dataset.
mnist = tf.keras.datasets.mnist

# setup the data
# Convert the samples from integers to floating point numbers
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Classifier
# Build tf.keras model by stacking layers.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Loss function —This measures how accurate the model is during training.

# Optimizer —This is how the model is updated based on the data it sees and its loss function

# Select an optimizer and loss function used for training
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training and Evaluate model
model.fit(x_train, y_train, epochs=5)

# Evalute
model.evaluate(x_test, y_test)

test_loss, test_acc = model.evaluate(x_test, y_test)

# print the Accuracy by evaluating the model
print('Test accuracy:', test_acc)