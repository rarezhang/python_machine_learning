"""
python implementation of Sequential Backward Selection (SBS) algorithm
"""

import numpy as np 
from itertools import combinations 

from sklearn.base import clone 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 


class SBS():
    """
    sequential backward selection 
    """
    
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        """
        
        :para estimator: classifier
        :para k_features: specify the desired number of features to return 
        :para scoring: evaluate the performance of a model 
        :para test_size:
        :random_state:
        """        
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.scoring = scoring
        self.test_size = test_size
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        
        dim = X_train.shape[1]  # number of total features 
        self.indices_ = tuple(range(dim)) # column indices of the final feature subset --> can use via the transform method to return a new data array with the selected feature columns
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_) 
        self.scores_ = [score]
        
        while dim > self.k_features:  # until the feature subset has the desired dimensionality
            scores = []
            subsets = []
            
            # combinations: create the feature subsets
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
                
            best = np.argmax(scores)
            self.indices_ = subsets[best]  # column indices of the final feature subset 
            self.subsets_.append(self.indices_)
            dim -= 1
            
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]    
        return self 
    
    def transform(self, X):
        """
        """
        # self.indices_ --> feature subset 
        return X[:, self.indices_]
        
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        """
        """
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
