import numpy as np
import pandas as pd
from math import exp
import matplotlib.pyplot as plt

def sigmoid_activation(training_data, training_output, weights, alpha, epochs):
    errors = []
    epoch_num = 0
    for epoch in range(epochs):
        error_sum = 0
        epoch_num += 1
        for i in range(len(training_data)):
            x = training_data[i]
            y = training_output[i]
            pred = np.dot(weights, x)
            output = 1 / (1 + exp(-pred))
            error = y - output  # Corrected error calculation
            for j in range(len(weights)):
                weights[j] += alpha * error * x[j]
            error_sum += error ** 2
        errors.append(error_sum)
        if error_sum <= 0.002:
            return weights, errors, epoch_num
    return weights, errors, epoch_num


def test_perceptron(weights, test_data,test_output):
    correct_predictions = 0
    total_predictions = len(test_data)

    for i in range(len(test_data)):
        x=test_data[i]
        y=test_output[i]
        pred = np.dot(weights, x)
        output = 1 / (1 + exp(-pred))
        if round(output) == y:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

data = pd.read_excel("bill.xlsx")
data.drop(columns=["Customer"], axis=1, inplace=True)
x = ((data.drop(columns=['High Value']).values - np.mean(data.drop(columns=['High Value']).values)) / np.std(data.drop(columns=['High Value']).values))
y = data['High Value'].apply(lambda x: 1 if x == "yes" else 0).values

weights = [0.5, 0.5, 0.5 , 0.5]

trained_weights, errors, epoch_num = sigmoid_activation(x, y, weights, 0.5, 1000)

print("Sigmoid activation")
print(f"Finalized weights are W0:{trained_weights[0]}, W1:{trained_weights[1]},W2:{trained_weights[2]}, W4:{trained_weights[3]}")
print(f"number of Epoches needed is {epoch_num}")
plt.plot(range(len(errors)), errors)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('sigmoid activation')
plt.show()

accuracy=test_perceptron(weights,x,y)
print(f"Accuracy: {accuracy}, Epochs: {epoch_num}")