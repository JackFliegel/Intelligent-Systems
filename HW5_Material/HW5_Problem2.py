import numpy as np
import random
# from itertools import count
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

def save_weights_to_file(weights, file_name):
    with open(file_name, 'w') as file:
        for row in weights:
            line = ' '.join(map(str, row))
            file.write(line + '\n')

class SOFM:
    # This class defines a self organizing feature map

    def __init__(self, input_size, grid_size):
        self.grid_size = grid_size
        self.weights = np.random.random((grid_size * grid_size, input_size))

    def winner(self, input_vector):
        # Finds the neuron (weight vector) closest to the input vector.
        distances = self.distance(self.weights, input_vector)
        return np.argmin(distances) # Indexes of min values along axis

    def update_weights(self, winner_idx, input_vector, learning_rate, sigma):
        winner_x, winner_y = divmod(winner_idx, self.grid_size)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                idx = x * self.grid_size + y
                distance = (winner_x - x) ** 2 + (winner_y - y) ** 2
                influence = np.exp(-distance / (2 * (sigma ** 2)))
                self.weights[idx] += learning_rate * influence * (input_vector - self.weights[idx])

    def distance(self, matrix, vector):
        # Calculates the Euclidean distance between a matrix of vectors and a single vector
        return np.sqrt(np.sum((matrix - vector) ** 2, axis=1))

def train_sofm(sofm, data, learning_rate=0.3, sigma=1.0, epochs=2):
    # Trains the SOFM on the provided data for a given number of epochs
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        for input_vector in data:
            winner_idx = sofm.winner(input_vector) #  finds the winner neuron for each input
            sofm.update_weights(winner_idx, input_vector, learning_rate, sigma) # updates the weights accordingly

def test_sofm(sofm, data, labels):
    # Tests SOFM using test data and corresponding labels
    activity_matrix = np.zeros((10, sofm.grid_size, sofm.grid_size))
    for input_vector, label in zip(data, labels):
        winner_idx = sofm.winner(input_vector)
        winner_x, winner_y = divmod(winner_idx, sofm.grid_size)
        activity_matrix[label][winner_x][winner_y] += 1
    return activity_matrix / 100  # Normalize by number of data points per class

def plot_heatmaps(activity_matrices):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flatten()):
        heatmap = ax.imshow(activity_matrices[i], cmap='afmhot', interpolation='nearest')
        ax.set_title(f'Class {i}')
        fig.colorbar(heatmap, ax=ax)
    plt.show()

def create_image(weights):
    # Array -> image shape
    return weights.reshape(28, 28)

def plot_weight_grids(sofm):
    # Plots a 12x12 grid of images representing the weights of each neuron in the SOFM.

    grid_size = sofm.grid_size
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    for i in range(grid_size):
        for j in range(grid_size):
            neuron_index = i * grid_size + j
            neuron_weights = sofm.weights[neuron_index]
            image = create_image(neuron_weights)

            ax = axes[i, j]
            ax.imshow(image, cmap='gray')
            ax.axis('off')

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

    random.shuffle(trainingSetPixels)
    # Initialize and train SOFM
    sofm = SOFM(input_size=784, grid_size=12)
    train_sofm(sofm, trainingSetPixels)

    # Test SOFM
    activity_matrices = test_sofm(sofm, testSetPixels, testSetTargets)

    # Plot figures
    save_weights_to_file(sofm.weights, './HW5_datafiles/sofm_weights.txt')
    plot_heatmaps(activity_matrices)
    plot_weight_grids(sofm)

    return sofm.weights # return weights to be used in problem 3
