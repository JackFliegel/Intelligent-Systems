from HW5_Problem1 import *
import numpy as np
import random


def runNetwork(weights):
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

    def load_pretrained_weights(file_path):
        with open(file_path, 'r') as file:
            weights = np.array([list(map(float, line.strip().split())) for line in file])
        return weights

    def train(layer1, activation1, layer2, activation2, isTrainInputs):
        train_accuracies, val_accuracies, train_errors, val_errors = [], [], [], []  # Lists to save errors at every 10th epoch
        for epoch in range(epochs): # Training loop
            # Create batch
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
            if isTrainInputs:
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
        return train_accuracies, val_accuracies, train_errors, val_errors

        

    # CASE 1 ------------------------------------------------------------------------------------------------------------------------
    
    layer1Case1 = Layer(784, 144, weights)# 784 'features' and 250 output to hidden layer, load weights
    activation1Case1 = Activation()

    layer2Case1 = Layer (144, 10) #hidden layer to 10 outputs
    activation2Case1 = ActivationSoftmax()
    
    train_accuracies_caseI, val_accuracies_caseI, train_errors_caseI, val_errors_caseI = train(layer1Case1, activation1Case1, layer2Case1, activation2Case1, isTrainInputs=False)

    # plot confusion matrix
    # Forward pass testSet
    layer1Case1.forward(np.array(testSetPixels))
    activation1Case1.forward(layer1Case1.output)
    layer2Case1.forward(activation1Case1.output)
    activation2Case1.forward(layer2Case1.output)
    plot_confusion_matrix(compute_confusion_matrix(testSetTargets, calculate_predictions(activation2Case1.output)), title='Test Set Case I')
    