"""
python implementation of the Majority voting classifier 

more sophisticated version: sklearn.ensemble.VotingClassifier
"""

# use the parent classes BaseEstimator and ClassifierMixin to get some base functionality, e.g., get_params, set_params
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from sklearn.preprocessing import LabelEncoder
# import six to make the MajorityVoteClassifier compatible with Python 2.7
from sklearn.externals import six
from sklearn.pipeline import _name_estimators

import numpy as np
import operator



class MajorityVoteClassifier(BaseEstimator, ClassifierMixin): 
    """
    a majority vote ensemble classifier 
    """
    
    def __init__(self, classifiers, vote='classlabel', weights=None):
        """
        
        :para classifiers: array-like shape=[n_classifiers], different classifiers for the ensemble  
        
        :para vote: str, {'classlabel', 'probability'}, default: 'classlabel'
            - 'classlabel': the prediction is based on the argmax of class labels
            - 'probability': the argmax of the sum of probabilities is used to predict the class label (recommended for calibrated classifiers)
        
        :para weights: array-like, shape=[n_classifiers], default=None  
            - None: uniform weights  
            - list of `int` or `float` values: classifiers are weighted by importance  
        """
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote 
        self.weights = weights  
        
    def fit(self, X, y):
        """
        fit classifiers. 
        
        :para X: {array-like, sparse matrix}, shape=[n_samples, n-features]
            matrix of training samples 
        
        :para y: array-lake, shape=[n_samples]  
            vector of target class labels  
            
        :return self: object
        """
        # use LabelEncoder to ensure class labels start with 0
        # important for np.argmax call in self.predict 
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self     
        
    def predict(self, X):
        """
        predict class labels for X
        
        :para X: {array-like, sparse matrix}, shape=[n_samples, n_features]
            matrix of testing samples 
            
        :return maj_vote: array-like, shape=[n_samples] 
            predicted class labels 
        """
        # 'probability' vote
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        # 'classlabel' vote
        else:
            # collect results from clf.predict calls 
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, weights=self.weights)), # function: accept 1-D arrays. It is applied to 1-D slices of arr along the specified axis.
            axis=1, # Axis along which arr is sliced.
            arr=predictions)  # input array: numpy array.
        maj_vote = self.lablenc_.inverse_transform(maj_vote)  # from 0,..,1 back to original labels 
        return maj_vote
        
    def predict_proba(self, X):
        """
        predict class probabilities for X 
        
        :para X: {array-like, sparse matrix}, shape=[n_samples, n_features]
            training vectors
            n_samples: the number of samples  
            n_features: the number of features  
        
        :return avg_proba: array-like, shape=[n_samples, n_classes]
            weighted average probability for each class per sample  
        """
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba
        
    def get_params(self, deep=True):
        """
        get classifier parameter names for GridSearch
        useful for grid search for hyperparameter-tuning
        access the parameters of individual classifiers in the ensemble
        """
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out 
            