import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import deepNeuralNetwork

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess data: Flatten images and normalize pixel values
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# One-hot encode labels
y_train = tf.one_hot(y_train, depth=10).numpy()
y_test = tf.one_hot(y_test, depth=10).numpy()

# Define the size of the neural network [input, hidden_layer, output]
size = [x_train.shape[1], 128, 10]

# Initialize the neural network
dnn = deepNeuralNetwork(size, activation="relu")

# Train the neural network
dnn.train(x_train, y_train, x_test, y_test, epochs=10, batch_size=64, optimizer="adam")

# Evaluate the model on the test set
test_output = dnn.forward_prop(x_test)
test_accuracy = dnn.accuracy(y_test, test_output)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Visualize test results
def visualize_results(x_test, y_test, y_pred, num_samples=10):
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {np.argmax(y_test[i])}, Pred: {np.argmax(y_pred[:, i])}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_results(x_test, y_test, test_output)
