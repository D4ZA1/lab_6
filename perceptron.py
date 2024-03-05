import numpy as np
import matplotlib.pyplot as plt
from math import exp


def step_activation(x):
    return x>0

def sigmoid_activation(x):
    return 1/(1+exp(-x))

def bi_polar_activation(x):
    if x>0:
        return 1
    elif x==0:
        return 0
    else:
        return -1

def reLU_activation(x):
    if x>0:
        return x
    else:
        return 0


training_data = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
W0, W1, W2 = 10, 0.2, -0.75
alpha = 0.05
epochs = 1000
errors = []
for epoch in range(epochs):
    error_sum = 0
    for x in training_data:
        output = step_activation(W0*x[0] + W1*x[1] + W2*1)
        error = x[2] - output
        W0 += alpha * error * x[0]
        W1 += alpha * error * x[1]
        W2 += alpha * error * 1
        error_sum += error**2
    errors.append(error_sum)
    if error_sum<=0.002:
        break

# Plotting
print(f"Finalized weights are W0:{W0}, W1:{W1},W2:{W2}")
plt.plot(range(len(errors)), errors)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Error vs. Epochs')
plt.show()

