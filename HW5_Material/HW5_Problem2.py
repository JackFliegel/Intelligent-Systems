import os
import ast
import numpy as np
import random
from itertools import count
import matplotlib.pyplot as plt

def read_Labels():
    result = []
    # Reads in the label values (Will create the different data sets from this)
    with open('./HW5_datafiles/MNISTnumLabels5000_balanced.txt', 'r') as file:
        for line_number, line in enumerate(file, start=1):
            result.append([line_number, int(line.strip())])
    return result
    
def read_Pixels():
    pixels = []
    # Reads in the pixel values 
    with open('./HW5_datafiles/MNISTnumImages5000_balanced.txt', 'r') as file:
        for currentStep, line in enumerate(file, start=0):
            pixels.append([float(num) for num in line.strip().split()]) #adds the values all 784 pixels into array (indexed: 0 to 783)
    return pixels

def writeSet(data, str):    
    with open(f'HW5_datafiles/{str}', 'w') as file:
        for item in data:
            file.write(f'{item}\n')

class Neuron:
    def __init__(self, inputs) -> None:
        self.inputs = inputs
        self.weights = []
        self.bias = 0 # bias starts at 0
        self.outputs = []

class Layer:
    def __init__(self, n_inputs, n_neurons, weights=None) -> None:
        if weights is None:
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # init weights
        else:
            self.weights = weights.T
            print(self.weights.shape)
        self.biases = np.zeros((1, n_neurons)) # zero out the init biases
        # Initialize momentum terms
        # self.prev_dweights = np.zeros_like(self.weights)
        self.prev_dweights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.prev_dbiases = np.zeros_like(self.biases)
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = np.array(inputs)

    def backward(self, dvalues):
        #Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues) #transpose
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        #Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)  
        self.inputs = inputs    

    def backward(self, dvalues):
        self.dinputs = np.array(dvalues, copy=True)
        self.dinputs[self.inputs <= 0] = 0

class ActivationSoftmax:
    def forward(self, inputs):
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 
        probabilities = expValues / np.sum(expValues, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    def calculate(self, output, y):
        sampleLosses = self.forward(output, y)
        dataLoss = np.mean(sampleLosses)
        return dataLoss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, yPred, yTrue):
        yTrue = np.array(yTrue)
        samples = len(yPred)
        yPred_clipped = np.clip(yPred, 1e-7, 1-1e-7)

        if len(yTrue.shape) == 1:
            correctConfidences = yPred_clipped[range(samples), yTrue]
        if len(yTrue.shape) == 2: #One hot encoded vectors
            correctConfidences = np.sum(yPred_clipped*yTrue, axis=1)

        negative_log_likelihoods = -np.log(correctConfidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, yTrue):
        yTrue = np.array(yTrue) # Convert yTrue to a NumpPy array
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(yTrue.shape) == 1:
            yTrue = np.eye(labels)[yTrue]

        self.dinputs = -yTrue / dvalues
        self.dinputs = self.dinputs / samples

class Loss_J2(Loss):
    def forward(self, y_pred, y_true):
        y_true_adj = np.full_like(y_pred, 0.25) # set all to 0.25
        y_true_adj[np.arange(len(y_true)), y_true] = 0.75 # Set the target
        self.output = (y_pred - y_true_adj) ** 2
        return np.mean(self.output, axis=-1) # mean squared

    def backward(self, dvalues, y_true):
        y_true_adj = np.full_like(dvalues, 0.25) # set all to 0.25
        y_true_adj[np.arange(len(y_true)), y_true] = 0.75 # Set the target
        samples = len(dvalues)
        self.dinputs = 2 * (dvalues - y_true_adj) / samples

def calculate_accuracy(y_pred, y_true):
    y_true = np.array(y_true)
    predictions = np.argmax(y_pred, axis=1) # Winner takes all 
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1) # Winner takes all  
    accuracy = np.mean(predictions == y_true)
    return accuracy

def calculate_predictions(y_pred, threshold = 0.5):
    return np.argmax(y_pred, axis=1) # Winner takes all 

# --------------------- PLOT GRAPHS -----------------------------------
def plot_accuracy(train_accuracies, val_accuracies, epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(epochs), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')
    plt.show()

def compute_confusion_matrix(true_labels, predicted_labels, num_classes=10):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    for i in range(len(true_labels)):
        confusion_matrix[true_labels[i]][predicted_labels[i]] += 1
        
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, title='Confusion Matrix'):
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap='viridis')
    plt.title(title)
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_error_fraction(train_errors, val_errors, epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, epochs, 10), train_errors, label='Train Error Fraction')
    plt.plot(range(0, epochs, 10), val_errors, label='Validation Error Fraction')
    plt.xlabel('Epochs')
    plt.ylabel('Error Fraction')
    plt.legend()
    plt.title('Error Fraction over epochs')
    plt.show()

def save_weights_to_file(weights, file_name):

    with open(file_name, 'w') as file:
        for row in weights:
            line = ' '.join(map(str, row))
            file.write(line + '\n')

class SOFM:
    def __init__(self, input_size, grid_size):
        self.grid_size = grid_size
        self.weights = np.random.random((grid_size * grid_size, input_size))

    def find_winner(self, input_vector):
        distances = self.euclidean_distance(self.weights, input_vector)
        return np.argmin(distances)

    def update_weights(self, winner_idx, input_vector, learning_rate, sigma):
        winner_x, winner_y = divmod(winner_idx, self.grid_size)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                idx = x * self.grid_size + y
                distance = (winner_x - x) ** 2 + (winner_y - y) ** 2
                influence = np.exp(-distance / (2 * (sigma ** 2)))
                self.weights[idx] += learning_rate * influence * (input_vector - self.weights[idx])

    def euclidean_distance(self, matrix, vector):
        # Calculates the Euclidean distance between a matrix of vectors and a single vector
        return np.sqrt(np.sum((matrix - vector) ** 2, axis=1))

def train_sofm(sofm, data, learning_rate=0.1, sigma=1.0, epochs=5):
    for epoch in range(epochs):
        print(epoch)
        for input_vector in data:
            winner_idx = sofm.find_winner(input_vector)
            sofm.update_weights(winner_idx, input_vector, learning_rate, sigma)

def test_sofm(sofm, data, labels):
    activity_matrix = np.zeros((10, sofm.grid_size, sofm.grid_size))
    for input_vector, label in zip(data, labels):
        winner_idx = sofm.find_winner(input_vector)
        winner_x, winner_y = divmod(winner_idx, sofm.grid_size)
        activity_matrix[label][winner_x][winner_y] += 1
    return activity_matrix / 100  # Normalize by the number of data points per class

def plot_heatmaps(activity_matrices):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flatten()):
        heatmap = ax.imshow(activity_matrices[i], cmap='hot', interpolation='nearest')
        ax.set_title(f'Class {i}')
        fig.colorbar(heatmap, ax=ax)
    plt.show()
    
def runNetwork():
    data = read_Labels()
    pixels = read_Pixels()

    trainingSet, trainingSetTargets, trainingSetPixels = [], [], []
    for i in range(0, 5000, 500):
        trainingSet.extend(data[i:i+400])
        trainingSetTargets.extend(pair[1] for pair in data[i:i+400])
        trainingSetPixels.extend(pixels[i:i+400])
        
    testSet, testSetTargets, testSetPixels = [], [], []
    for i in range(0, 5000, 500):
        testSet.extend(data[i:i+100])
        testSetTargets.extend(pair[1] for pair in data[i:i+100])
        testSetPixels.extend(pixels[i:i+100])
        
    train_accuracies, val_accuracies = [], []
    train_errors, val_errors = [], []  # Lists to save errors at every 10th epoch
    
    # Randomize training data sets
    random.shuffle(trainingSet)
    #Write sets to txt
    writeSet(trainingSet, 'trainingSet.txt')
    writeSet(testSet, 'testSet.txt')

    X = np.array(trainingSetPixels)
    y = np.array(trainingSetTargets)
    learning_rate = 0.2
    # learning_rate = 0.01
    epochs = 200
    batch_size = 400
    momentum = 0.94

    def load_pretrained_weights(file_path):
        with open(file_path, 'r') as file:
            weights = np.array([list(map(float, line.strip().split())) for line in file])
        return weights
    
    layer1 = Layer(784, 250) # 784 'features' and 250 output to hidden layer
    activation1 = Activation()

    layer2 = Layer (250, 10) #hidden layer to 10 outputs
    activation2 = ActivationSoftmax()

    # Initialize and train SOFM
    sofm = SOFM(input_size=784, grid_size=12)
    train_sofm(sofm, trainingSetPixels)

    # Test SOFM
    activity_matrices = test_sofm(sofm, testSetPixels, testSetTargets)

    # Plot heatmaps
    plot_heatmaps(activity_matrices)

runNetwork()