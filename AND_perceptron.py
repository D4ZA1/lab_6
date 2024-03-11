import numpy as np
import matplotlib.pyplot as plt
from math import exp


def step_activation(training_data,weights,alpha,epochs):
    errors = []
    epoch_num=0
    for epoch in range(epochs):
        error_sum = 0
        epoch_num+=1
        for x in training_data:
            pred=weights[0]*x[0] + weights[1]*x[1] + weights[2]*1
            output = pred>0
            error = x[2] - output
            weights[0] += alpha * error * x[0]
            weights[1] += alpha * error * x[1]
            weights[2] += alpha * error * 1
            error_sum += error**2
        errors.append(error_sum)
        if error_sum<=0.002:
            return weights,errors,epoch_num
    return weights,errors,epoch_num

def sigmoid_activation(training_data,weights,alpha,epochs):
    errors = []
    epoch_num=0
    for epoch in range(epochs):
        error_sum = 0
        epoch_num+=1
        for x in training_data:
            pred=weights[0]*x[0] + weights[1]*x[1] + weights[2]*1
            output = 1/(1+exp(-pred))
            error = x[2] - output
            weights[0] += alpha * error * x[0]
            weights[1] += alpha * error * x[1]
            weights[2] += alpha * error * 1
            error_sum += error**2
        errors.append(error_sum)
        if error_sum<=0.002:
            return weights,errors,epoch_num
    return weights,errors,epoch_num
   

def bi_polar_activation(training_data,weights,alpha,epochs):
    errors = []
    epoch_num=0
    for epoch in range(epochs):
        error_sum = 0
        epoch_num+=1
        for x in training_data:
            pred=weights[0]*x[0] + weights[1]*x[1] + weights[2]*1
            if pred > 0:
                output = 1
            elif pred == 0:
                output = 0
            else:
                output = -1
            error = x[2] - output
            weights[0] += alpha * error * x[0]
            weights[1] += alpha * error * x[1]
            weights[2] += alpha * error * 1
            error_sum += error**2
        errors.append(error_sum)
        if error_sum<=0.002:
            return weights,errors,epoch_num
    return weights,errors,epoch_num


def reLU_activation(training_data,weights,alpha,epochs):
    errors = []
    epoch_num=0
    for epoch in range(epochs):
        error_sum = 0
        epoch_num+=1
        for x in training_data:
            pred=weights[0]*x[0] + weights[1]*x[1] + weights[2]*1
            if pred>0:
                output = pred
            else:
                output = 0
            error = x[2] - output
            weights[0] += alpha * error * x[0]
            weights[1] += alpha * error * x[1]
            weights[2] += alpha * error * 1
            error_sum += error**2
        errors.append(error_sum)
        if error_sum<=0.002:
            return weights,errors,epoch_num
    return weights,errors,epoch_num


def test_perceptron(weights, test_data, activation_function):
    correct_predictions = 0
    total_predictions = len(test_data)

    for x in test_data:
        pred = weights[0] * x[0] + weights[1] * x[1] + weights[2] * 1
        if activation_function == 'step':
            output = int(pred > 0)
        elif activation_function == 'sigmoid':
            output = 1 / (1 + exp(-pred))
        elif activation_function == 'bi_polar':
            if pred > 0:
                output = 1
            elif pred == 0:
                output = 0
            else:
                output = -1
        elif activation_function == 'reLU':
            output = max(0, pred)

        if round(output) == x[2]:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

training_data = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
weights = [10, 0.2, -0.75]
alpha = 0.05
epochs = 1000

SWeights,SErrors,SEpoch_num=step_activation(training_data,weights,alpha,epochs)

# Plotting
print("Step_activation")
print(f"Finalized weights are W0:{SWeights[0]}, W1:{SWeights[1]},W2:{SWeights[2]}")
print(f"number of Epoches needed is {SEpoch_num}")
plt.plot(range(len(SErrors)), SErrors)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('step activation')
plt.show()

Bweights,Berrors,Bepoch_num=bi_polar_activation(training_data,weights,alpha,epochs)

# Plotting
print("Bi-polar activation")
print(f"Finalized weights are W0:{Bweights[0]}, W1:{Bweights[1]},W2:{Bweights[2]}")
print(f"number of Epoches needed is {Bepoch_num}")
plt.plot(range(len(Berrors)), Berrors)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Bi-polar activation')
plt.show()

SiWeights,SiErrors,SiEpoch_num=sigmoid_activation(training_data,weights,alpha,epochs)

# Plotting
print("Sigmoid activation")
print(f"Finalized weights are W0:{SiWeights[0]}, W1:{SiWeights[1]},W2:{SiWeights[2]}")
print(f"number of Epoches needed is {SiEpoch_num}")
plt.plot(range(len(SiErrors)), SiErrors)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('sigmoid activation')
plt.show()

RWeights,RErrors,REpoch_num=reLU_activation(training_data,weights,alpha,epochs)

# Plotting
print("reLU activation")
print(f"Finalized weights are W0:{RWeights[0]}, W1:{RWeights[1]},W2:{RWeights[2]}")
print(f"number of Epoches needed is {REpoch_num}")
plt.plot(range(len(RErrors)), RErrors)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('reLU activation')
plt.show()

activation_functions = ['step', 'sigmoid', 'bi_polar', 'reLU']

for activation_function in activation_functions:
    accuracy = test_perceptron(weights, training_data, activation_function)
    print(f"Activation function: {activation_function}, Accuracy: {accuracy*100}")





# question 3
rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
relu_iter=[]
bi_iter=[]
si_iter=[]
st_iter=[]

for i in rate:
    _, _, rlepoch_num = reLU_activation(training_data, weights, i, epochs)
    _, _, SiEpoch_num = sigmoid_activation(training_data, weights, i, epochs)
    _ , _,Bepoch_num = bi_polar_activation(training_data, weights, i, epochs)
    _,ws, SEpoch_num = step_activation(training_data, weights, i, epochs)
    relu_iter.append(rlepoch_num)
    si_iter.append(SiEpoch_num)
    bi_iter.append(Bepoch_num)
    st_iter.append(SEpoch_num)

plt.plot(rate,relu_iter)
plt.title('reLU activation')
plt.xlabel('learning rate')
plt.ylabel('Epoches')
plt.show()

plt.plot(rate,st_iter)
plt.title('Step activation activation')
plt.xlabel('learning rate')
plt.ylabel('Epoches')
plt.show()

plt.plot(rate,si_iter)
plt.title('sigmoid activation')
plt.xlabel('learning rate')
plt.ylabel('Epoches')
plt.show()

plt.plot(rate,bi_iter)
plt.title('Bi-polar activation')
plt.xlabel('learning rate')
plt.ylabel('Epoches')
plt.show()
