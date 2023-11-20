import numpy as np
import random
import ast
from itertools import count
import matplotlib.pyplot as plt

def read_Labels():
    result = []
    # Reads in the label values (Will create the different data sets from this)
    with open('./HW3_datafiles/MNISTnumLabels5000_balanced.txt', 'r') as file:
        for line_number, line in enumerate(file, start=1):
            result.append([line_number, int(line.strip())])
        return result
    
def read_Pixels():
    pixels = []
    # Reads in the pixel values 
    with open('./HW3_datafiles/MNISTnumImages5000_balanced.txt', 'r') as file:
        for currentStep, line in enumerate(file, start=0):
            pixels.append([float(num) for num in line.strip().split()]) #adds the values all 784 pixels into array (indexed: 0 to 783)
    return pixels

def writeSet(data, str):
    if str == 'trainingSet':
        setType = 'trainingSet.txt'
    elif str == 'challengeSet':
        setType = 'challengeSet.txt'
    elif str == 'testSet':
        setType = 'testSet.txt'
    
    with open(f'HW3_datafiles/{setType}', 'w') as file:
        for item in data:
            file.write(f'{item}\n')


def readSet(str):
    if str == 'trainingSet':
        setType = 'trainingSet.txt'
    elif str == 'challengeSet':
        setType = 'challengeSet.txt'
    elif str == 'testSet':
        setType = 'testSet.txt'
    elif str == 'initWeights':
        setType = 'initWeights.txt'

    toReturn = []

    with open(f'HW3_datafiles/{setType}', 'r') as file:
        for line in file:
            data = ast.literal_eval(line.strip())
            toReturn.append(data)
    return toReturn


class Perceptron:
    
    def __init__(self, learningRate, dataSet, pixelInput, alpha = 0.1):
        self.output = []
        self.weights = []
        self.N = learningRate
        self.dataSet = dataSet
        self.pixels = pixelInput

        self.errFract = 0
        self.precision = 0
        self.recall = 0
        self.F1 = 0
        self.specificity = 0

        # Creates a random weight for each pixel (28 x 28 image = 784)
        for j in range(0, 784):
            tempW = random.uniform(-0.5, 0.5)
            self.weights.append([j, tempW])
        self.w_0 = random.uniform(-0.5, 0.5) #initializes random weight for w_0

        #Writes the initial values of the perceptron to file
        with open('HW3_datafiles/initWeights.txt', 'w') as file:
            for item in self.weights:
                file.write(f'{item}\n')
    
    def changeSet(self, newDataSet):
        self.dataSet = newDataSet
        self.output = []

    def runSet(self):
        self.output = []
        for row in self.dataSet: #Goes through every row in set (represents the row of pixels to get)
            self.output.append([row[0], self.NetInput(row[0])]) # adds the output as [row of pixels, 1/0]

        return self.output
    
    def NetInput(self, x_row):
        x0 = 1 # bias input is always set to 1
        total = x0 * self.w_0 # Add in the bias input * bias weight
        for i in range(0, 784): #Sum inputs
            total += self.weights[i][1] * self.pixels[x_row][i]
        return self.step(total)
    
    def step(self, total):
        return 1 if total > 0 else 0
    
    def findOutputValue(self, key):
        for i in self.output:
            if i[0] == key:
                return i[1]
        return None
    
    def findActualOutput(self, key):
        for i in self.dataSet:
            if i[0] == key:
                return bool(i[1]) # Bool cast if i[1] is digits 2 - 9
        return None
    
    def updateWeights(self, rowNumber):
        weightVals = [values[1] for values in self.weights] # Extracts weight values into array
        predicted = self.findOutputValue(rowNumber)
        target = self.findActualOutput(rowNumber)
        # target = self.step(np.dot(weightVals, self.pixels[rowNumber]) + self.w_0)
        
        for i in range(len(weightVals)): #Runs through row and changes weights for row
            self.weights[i][1] += (self.N * (target - predicted) * self.pixels[rowNumber][i])
        self.w_0 += self.N * (target - predicted) # Changes bias weight

    
    def getErrors(self):
        truePos, trueNeg, falsePos, falseNeg = 0, 0, 0, 0
        for i in range(len(self.dataSet)):
                if bool(self.dataSet[i][1]) == bool(self.output[i][1])and self.output[i][1]: #true positive
                    truePos += 1
                elif bool(self.dataSet[i][1]) == bool(self.output[i][1])and not self.output[i][1]: #true negative:
                    trueNeg += 1
                elif bool(self.dataSet[i][1]) != bool(self.output[i][1])and self.output[i][1]: #false positive:
                    falsePos += 1
                elif bool(self.dataSet[i][1]) != bool(self.output[i][1])and not self.output[i][1]: #false negatives:
                    falseNeg += 1       
        self.precision = truePos/(truePos + falsePos + 1e-10)
        self.recall = truePos/(truePos + falseNeg + 1e-10)
        self.specificity = falsePos/(falsePos + trueNeg + 1e-10)
        self.F1 = 2 * ((self.precision * self.recall)/ (self.precision + self.recall + 1e-10))
        self.errFract = (falseNeg + falsePos) / (falseNeg + falsePos + trueNeg + truePos)
        if(True):
            print(f'truePos = {truePos} \ntrueNeg = {trueNeg} \nfalsePos = {falsePos}\nfalseNeg = {falseNeg}') 
            print(f'F1 = {self.F1: .3f}\nError Fraction = {self.errFract}\n')

    def trainSet(self):
        for pixelRow in self.dataSet: #Goes through every row in set (row of pixels to get)
            self.updateWeights(pixelRow[0])

    def train(self):
        print(f'Before Training:')
        self.runSet()
        self.getErrors()
        self.beforeMetrics = [self.precision, self.recall, self.F1, self.errFract]

        self.epochs = []
        self.errors = []

        for i in range(1, 40): # Loop that represents epochs
            print(f'Epoch {i}:')
            self.trainSet()

            self.runSet()
            self.getErrors()
            self.epochs.append(i)
            self.errors.append(self.errFract)

        #Writes the final values of the perceptron to file
        with open('HW3_datafiles/finalWeights.txt', 'w') as file:
            for item in self.weights:
                file.write(f'{item}\n')

    def plot(self):
        # Plotting
        plt.plot(self.epochs, self.errors, label = 'Epoch vs Error')
        plt.xlabel('Epoch')
        plt.ylabel('Error Fraction')
        plt.title('Perceptron Epoch vs Error')
        plt.legend()

        self.afterMetrics = [self.precision, self.recall, self.F1, self.errFract]
        labels = ['Precision', 'Recall', 'F1', 'Error Fraction']
        # Bar graph
        barGraph, ax = plt.subplots()
        barWidth = 0.35
        bar1 = ax.bar(np.arange(len(labels))-barWidth/2, self.beforeMetrics, barWidth, label='Before Training')
        bar2 = ax.bar(np.arange(len(labels))+barWidth/2, self.afterMetrics, barWidth, label='After Training')
        ax.set_ylabel('Scores')
        ax.set_title('Metrics Before and After Training')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.legend()

        plt.show()

    def plotBiasMetrics(self):
        errorFractions, precisions, recalls, F1s = [], [], [], []

        trainedBias = self.w_0
        lower = trainedBias * (1 - 0.2) # 20% below trained bias
        upper = trainedBias * (1 + 0.2) # 20% above trained bias
        biasWeights = np.linspace(lower, upper, 20)
        for biasWeight in biasWeights:
            self.w_0 = biasWeight
            self.runSet()
            self.getErrors()

            errorFractions.append(self.errFract)
            precisions.append(self.precision)
            recalls.append(self.recall)
            F1s.append(self.F1)

        # Plot bias metrics
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(biasWeights, errorFractions, label='Error Fractions')
        plt.plot(biasWeights, precisions, label='Precisions')
        plt.plot(biasWeights, recalls, label='Recalls')
        plt.plot(biasWeights, F1s, label='F1 Scores')
        plt.axvline(x=trainedBias, color='k', linestyle='--', label='Trained Bias Weight')
        plt.xlabel('Bias Weight (w0)')
        plt.ylabel('Metrics')
        plt.title('Bias Weight Sensitivity')
        plt.legend()

        plt.show()
    
    def plotHeatMap(self):
        init, final = [], []
        with open(f'HW3_datafiles/initWeights.txt', 'r') as file:
            init = [ast.literal_eval(line.strip()) for line in file]
        with open(f'HW3_datafiles/finalWeights.txt', 'r') as file:
            final = [ast.literal_eval(line.strip()) for line in file]
        
        # Extract values from the list of [index, value] pairs
        values_init = np.array(init)[:, 1].astype(float)
        values_final = np.array(final)[:, 1].astype(float)

        # Reshape the weights into 28x28 matrices
        weights_init_matrix = np.reshape(values_init, (28, 28))
        weights_final_matrix = np.reshape(values_final, (28, 28))

        # Plots weights side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(weights_init_matrix, cmap='inferno', interpolation='nearest')
        axes[0].set_title('Initial Weights')

        axes[1].imshow(weights_final_matrix, cmap='inferno', interpolation='nearest')
        axes[1].set_title('Final Weights')

        plt.show()

    def plotErrorValues(self):

        return 0
    
    def tableOutputs(self):
        count0s = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        count1s = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for data in self.dataSet:
            if self.findOutputValue(data[0]):
                count1s[data[1]] += 1
            else:
                count0s[data[1]] += 1

        data = count0s, count1s

        # Create a figure and hide axis
        fig, ax = plt.subplots()
        ax.axis('off')

        # Create the table
        table = ax.table(cellText=data, rowLabels=['0', '1'], colLabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], loc='center')

        # Set the table style
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)  # Adjust the table size if needed

        # Show the plot
        plt.show()





def main():
    data = read_Labels()
    pixels = read_Pixels()

    trainingSet = data[0:400] + data[500:900]
    trainingSet2 = data[0:100] + data[4500:4600]
    testSet = data[400:500] + data[900:1000]
    challengeSet = data[1000:1100] + data[1500:1600] + data[2000:2100] + data[2500:2600] + data[3000:3100] + data[3500:3600] + data[4000:4100] + data[4500:4600]
    challengeSet2 = data[500:600] + data[1000:1100] + data[1500:1600] + data[2000:2100] + data[2500:2600] + data[3000:3100] + data[3500:3600] + data[4000:4100]

    # Randomize training data sets
    random.shuffle(trainingSet)
    random.shuffle(trainingSet2)

    # --------------PROBELM 1--------------

    # Create Perceptron with training Set
    perceptron = Perceptron(learningRate=0.1, dataSet=trainingSet, pixelInput=pixels)

    # Train the perceptron (Probelm 1 parts 1 , 2, 3, and 4)
    perceptron.train()

    # Change data to test set
    perceptron.changeSet(testSet)
    perceptron.runSet()

    # Plot the results of perceptron (Problem 1 parts 5 and 6)
    perceptron.plot()

    # Plot the Bias metrics (Problem 1 part 7)
    perceptron.plotBiasMetrics()

    # Plot the Heatmap (Problem 1 part 8)
    perceptron.plotHeatMap()

    # Change to Challenge set, run perceptron, and print the outputs (Problem 1 part 9)
    perceptron.changeSet(challengeSet)
    perceptron.runSet()
    perceptron.tableOutputs()

    # --------------PROBELM 2--------------
    
    # Create new Perceptron with 0 and 9 data set
    perceptron2 = Perceptron(learningRate=0.1, dataSet=trainingSet2, pixelInput=pixels)

    perceptron2.train()

    perceptron2.plot()

    perceptron2.plotBiasMetrics()

    perceptron2.plotHeatMap()

    # Change to Challenge set
    perceptron2.changeSet(challengeSet2)
    perceptron2.runSet()
    perceptron2.tableOutputs()

main()