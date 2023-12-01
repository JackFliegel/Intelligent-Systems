from feedForward import read_Labels, read_Pixels
import numpy as np
import os
import matplotlib.pyplot as plt

class Autoencoder:
    def __init__(self, input_size, hidden_size, learning_rate, beta):
        # init weights
        self.weights = 0.01 * np.random.randn(hidden_size, input_size) #init weights
        self.biases = np.zeros((hidden_size, 1))
        self.prev_weights =  0.01 * np.random.randn(input_size, hidden_size) # init prev weights
        self.prev_biases = np.zeros((input_size, 1))

        # init momentum
        self.dinput_weights = np.zeros((hidden_size, input_size))
        self.dinput_biases = np.zeros((hidden_size, 1))
        self.dinput_prev_weights = np.zeros((input_size, hidden_size))
        self.dinput_prev_biases = np.zeros((input_size, 1))

        self.learning_rate = learning_rate
        self.beta = beta

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def activation(self, Z):
        return np.maximum(0, Z)

    def activation_derivative(self, Z):
        return (Z > 0).astype(float)

    def calculate_loss(self, y_true, y_pred):
        return 0.5 * np.mean(np.sum((y_true - y_pred) ** 2, axis=0))

    def forward(self, X):
        Z1 = np.dot(self.weights, X) + self.biases #linear transformation 1
        A1 = self.activation(Z1)
        Z2 = np.dot(self.prev_weights, A1) + self.prev_biases #linear transformation 2
        A2 = self.sigmoid(Z2)
        return Z1, A1, Z2, A2

    def backward(self, X, Y, Z1, A1, Z2, A2):
        batch_num = X.shape[1] 

        # gradient of the loss with respect to Z2
        dZ2 = A2 - Y 

        # Calculate gradients for the second (output) layer
        dprev_weights = np.dot(dZ2, A1.T) / batch_num
        dprev_biases = np.sum(dZ2, axis=1, keepdims=True) / batch_num

        # Derivative of loss with respect output of the first layer
        dZ1 = np.dot(self.prev_weights.T, dZ2) * self.activation_derivative(Z1)

        #Calculate weights/biases for first hidden layer
        dweights = np.dot(dZ1, X.T) / batch_num
        dbiases = np.sum(dZ1, axis=1, keepdims=True) / batch_num
        
        # Apply momentum
        self.dinput_weights = self.beta * self.dinput_weights + (1 - self.beta) * dweights
        self.dinput_biases = self.beta * self.dinput_biases + (1 - self.beta) * dbiases
        self.dinput_prev_weights = self.beta * self.dinput_prev_weights + (1 - self.beta) * dprev_weights
        self.dinput_prev_biases = self.beta * self.dinput_prev_biases + (1 - self.beta) * dprev_biases
        
        # Update weightss
        self.prev_weights -= self.learning_rate * self.dinput_prev_weights
        self.prev_biases -= self.learning_rate * self.dinput_prev_biases
    
    def train(self, X, Y, X_testSet, Y_testSet, epochs):
        X = np.array(X)
        Y = np.array(Y)
        X_testSet = np.array(X_testSet)
        Y_testSet = np.array(Y_testSet)

        training_loss, test_loss, epochs_list = [], [], []

        for epoch in range(epochs+1):
            # forward pass
            Z1, A1, Z2, A2_trainSet = self.forward(X) 
            loss = self.calculate_loss(X, A2_trainSet)
            self.backward(X, Y, Z1, A1, Z2, A2_trainSet) #updates weights + momentum

            _, _, _, A2_test = self.forward(X_testSet)
            loss_test = self.calculate_loss(Y_testSet, A2_test)

            if epoch % 10 == 0:
                epochs_list.append(epoch)
                training_loss.append(loss)
                test_loss.append(loss_test)
                os.system('cls')
                print(f'Epoch: {epoch}\nTraining Loss: {loss}------Test Loss: {loss_test}')
                
        return training_loss, test_loss, epochs_list

# Normalize the weights to be between 0 and 255
def normalize_image(image):
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val != 0:
        image = 255 * (image - min_val) / (max_val - min_val)
    else:
        image = 255 * image
    return image.astype(np.uint8)

# Plot features of neurons in grayscale
def plot_features(weights, neuron_index, subplot_index):
    if not (0 <= neuron_index < weights.shape[0]):
        raise ValueError(f"Neuron index {neuron_index} is out of bounds.")

    neuron_weights = weights[neuron_index, :]
    neuron_image = neuron_weights.reshape(28, 28)
    neuron_image = normalize_image(neuron_image)

    plt.subplot(5, 4, subplot_index)
    plt.imshow(neuron_image, cmap='gray')
    plt.title(f'Neuron #{neuron_index}')
    plt.colorbar(label='Weight Intensity')

def save_weights_to_file(weights, file_name):
    with open(file_name, 'w') as file:
        for row in weights:
            line = ' '.join(map(str, row))
            file.write(line + '\n')

def runAutoencoder():
    labels = read_Labels()
    pixels = read_Pixels()

    # Initialize
    epochs = 200
    input_size = 784  
    hidden_size = 250
    learning_rate = 0.2
    momentum = 0.94
    autoencoder = Autoencoder(input_size, hidden_size, learning_rate, momentum)
    
    trainingSet, trainingSetTargets, trainingSetPixels = [], [], []
    testSet, testSetTargets, testSetPixels = [], [], []
    for i in range(0, 5000, 500):
        #training set
        trainingSet.extend(pair[0] for pair in labels[i:i+400])
        trainingSetTargets.extend(pair[1] for pair in labels[i:i+400])
        trainingSetPixels.extend(pixels[i:i+400])

        #test set
        testSet.extend(pair[0] for pair in labels[i:i+100])
        testSetTargets.extend(pair[1] for pair in labels[i:i+100])
        testSetPixels.extend(pixels[i:i+100])

    input_train = np.array(trainingSetPixels).T
    input_test = np.array(testSetPixels).T
     
    # Train the autoencoder
    results = autoencoder.train(input_train, input_train, input_test, input_test, epochs)

    # reconstruct test images
    autoencoder.forward(input_test)

    # plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(results[2], results[0], label='Training Loss')
    plt.plot(results[2], results[1], label='Test Loss')
    plt.title('Training/Test Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # random_indicies = [199, 112, 49, 153, 72, 136, 30, 85, 14, 167, 98, 58, 191, 24, 120, 89, 177, 65, 150, 33]
    random_indicies = np.random.randint(0, 251, size=20)
    # 20 neruon subplots
    plt.figure(figsize=(12, 10))
    for i, neuron_index in enumerate(random_indicies, 1):
        plot_features(autoencoder.weights, neuron_index, i)
    plt.tight_layout()
    plt.show()

    save_weights_to_file(autoencoder.weights, 'pretrained_weights.txt')

runAutoencoder()