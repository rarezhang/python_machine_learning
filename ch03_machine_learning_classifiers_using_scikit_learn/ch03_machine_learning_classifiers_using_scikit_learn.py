"""
chapter 3
"""

import numpy as np 
from sklearn import datasets 
# from sklearn.cross_validation import train_test_split  # deprecation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import Perceptron 
from sklearn.metrics import accuracy_score

from utils import *   # plot_decision_regions 

# P 76 
# import data from sklearn 
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # numpy array 
y = iris.target  # numpy array 

# split the datasets into separate training ad test data sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# standardize the features 
sc = StandardScaler()
sc.fit(X_train)  # fit: estimated the parameters: sample mean & standard deviation 
X_train_std = sc.transform(X_train)  # use the transform method to standardized the training and testing data  
X_test_std = sc.transform(X_test)

# train a perceptron model 
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)  # eta0: learning rate, random_state: the initial shuffling of the training dataset after each epoch  
ppn.fit(X_train_std, y_train)

# make predictions 
y_pred = ppn.predict(X_test_std)
print(f'Misclassified samples: {(y_test != y_pred).sum()}')
# calculate the classification accuracy 
print(f'accuracy: {accuracy_score(y_test, y_pred)}')


# plot the result: specify the indices of the samples 
# P 79
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105,150))
plt.xlabel('petal length [standardized]') 
plt.ylabel('petal width [standardized]') 
plt.legend(loc='upper left')
plt.show()