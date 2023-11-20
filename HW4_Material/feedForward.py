import os
import numpy as np
import random
from itertools import count
import matplotlib.pyplot as plt

def read_Labels():
    result = []
    # Reads in the label values (Will create the different data sets from this)
    with open('./HW4_datafiles/MNISTnumLabels5000_balanced.txt', 'r') as file:
        for line_number, line in enumerate(file, start=1):
            result.append([line_number, int(line.strip())])
    return result
    
def read_Pixels():
    pixels = []
    # Reads in the pixel values 
    with open('./HW4_datafiles/MNISTnumImages5000_balanced.txt', 'r') as file:
        for currentStep, line in enumerate(file, start=0):
            pixels.append([float(num) for num in line.strip().split()]) #adds the values all 784 pixels into array (indexed: 0 to 783)
    return pixels

def writeSet(data, str):    
    with open(f'HW4_datafiles/{str}', 'w') as file:
        for item in data:
            file.write(f'{item}\n')

class Neuron:
    def __init__(self, inputs) -> None:
        self.inputs = inputs
        self.weights = []
        self.bias = 0 # bias starts at 0
        self.outputs = []

class Layer:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # init weights
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

def runFeedForward():
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
    epochs = 200
    batch_size = 400
    momentum = 0.94
    
    layer1 = Layer(784, 250) # 784 'features' and 150 output to hidden layer
    activation1 = Activation()

    layer2 = Layer (250, 10) #hidden layer to 10 outputs
    activation2 = ActivationSoftmax()

    for epoch in range(epochs):
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(0, len(X_shuffled), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]


        # Forward pass
        layer1.forward(X_batch)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)

        # lossFunction = Loss_CategoricalCrossentropy()
        lossFunction = Loss_J2()
        # Calculate loss and accuracy
        loss = lossFunction.calculate(activation2.output, y_batch)
        accuracy = calculate_accuracy(activation2.output, y_batch)

        # Backward pass
        lossFunction.backward(activation2.output, y_batch)
        activation2.backward(lossFunction.dinputs)
        layer2.backward(activation2.dinputs)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)

        # Update weights and biases
        layer1.prev_dweights = learning_rate * layer1.dweights + momentum * layer1.prev_dweights
        layer1.prev_dbiases = learning_rate * layer1.dbiases + momentum * layer1.prev_dbiases
        layer1.weights -= layer1.dweights
        layer1.biases -= layer1.dbiases
        layer2.prev_dweights = learning_rate * layer2.dweights + momentum * layer2.prev_dweights
        layer2.prev_dbiases = learning_rate * layer2.dbiases + momentum * layer2.prev_dbiases
        layer2.weights -= layer2.dweights
        layer2.biases -= layer2.dbiases

        # Validation
        layer1.forward(X_batch)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)

        # Calculate loss and accuracy for validation
        val_loss = lossFunction.calculate(activation2.output, y_batch)
        val_accuracy = calculate_accuracy(activation2.output, y_batch)

        # Save accuracies
        train_accuracies.append(accuracy)
        val_accuracies.append(val_accuracy)

        # Save Errors
        if (epoch + 1) % 10 == 0:  # Every 10th epoch
            train_error = 1 - accuracy
            val_error = 1 - val_accuracy

            train_errors.append(train_error)
            val_errors.append(val_error)
        
        # Print the progress
        os.system('cls')
        print(f'Epoch: {epoch+1}/{epochs}\nTraining Loss: {loss:.4f}')

    # plot the accuracy after training
    plot_accuracy(train_accuracies, val_accuracies, epochs)

    # plot the error fraction
    plot_error_fraction(train_errors, val_errors, epochs)

    # plot confusion matrix
    # Forward pass trainingSet
    layer1.forward(np.array(trainingSetPixels))
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    plot_confusion_matrix(compute_confusion_matrix(trainingSetTargets, calculate_predictions(activation2.output)))
    # Forward pass testSet
    layer1.forward(np.array(testSetPixels))
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    plot_confusion_matrix(compute_confusion_matrix(testSetTargets, calculate_predictions(activation2.output)))

