from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np


# MLP for AND gate and XOR gate
training_data_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_xor = np.array([0, 1, 1, 0])
training_data_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_and= np.array([0, 0, 0, 1])
alpha = 0.05
clf_xor = MLPClassifier(solver='sgd', learning_rate_init=alpha, hidden_layer_sizes=(4,), random_state=1)
#fit the model
clf_xor.fit(training_data_xor, output_xor)
X = clf_xor.predict([[0, 1], [1, 1],[0,0],[1,0]])
clf_and = MLPClassifier(solver='sgd', learning_rate_init=alpha, hidden_layer_sizes=(5,), random_state=1)
#fit the model
clf_and.fit(training_data_and, output_and)
Y = clf_and.predict([[0, 1], [1, 1],[0,0],[1,0]])

print(f"output for XOR is : {X}")
print(f"output for AND is : {Y}")

# MLP for data set
data= np.loadtxt("kinematic_features.txt")
X = data
y = np.concatenate((np.zeros(41), np.ones(55)))

#split the data into training and testing randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=41)
clf = MLPClassifier(solver='sgd', learning_rate_init=alpha, random_state=1)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(clf.score(X_test,y_test))