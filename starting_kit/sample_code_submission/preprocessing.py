"""
Author : Green
Last revision date : 03/04/2020
Description : File for Preprocessing
Revision History :
#CY : Ajout de tests
"""

# from preprocess import Preprocessor
from sys import argv
import warnings
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.base import BaseEstimator
    from zDataManager import DataManager # The class provided by binome 1
    # Note: if zDataManager is not ready, use the mother class DataManager
    from sklearn.decomposition import PCA
    from sklearn import preprocessing
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.feature_selection import VarianceThreshold

class Preprocessor(BaseEstimator):
    def __init__(self):
        # Pour avoir la 2D
        self.transformer = PCA(n_components=2)

    def fit(self, X, y=None):
        if X.ndim>1: self.num_feat = X.shape[1]
        if y.ndim>1: self.num_labels = y.shape[1]

        X_preprocess = self.preprocess.fit_transform(X)
        X_scaled = preprocessing.scale(X)
        self.mod.fit(X_preprocess, y)
        self.is_trained = True
        self.transformer = IsolationForest(random_state=0).fit(X_scaled)
        return self.transformer.fit(X, y)

    def predict(self, X):
        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        y = np.zeros([num_test_samples, self.num_labels])


        X_preprocess = self.preprocess.transform(X)
        y = self.mod.predict(X_preprocess)
        return y

    def save(self, path="./"):
        pass

    def load(self, path="./"):
        pass

    def fit_transform(self, X, y=None):
        return self.transformer.fit_transform(X)

    def transform(self, X, y=None):
        return self.transformer.transform(X)

class BasicClassifier(BaseEstimator):
    def __init__(self):
        '''This method initializes the parameters. This is where you could replace
        RandomForestClassifier by something else or provide arguments, e.g.
        RandomForestClassifier(n_estimators=100, max_depth=2)'''
        self.clf = RandomForestClassifier(random_state=1)

    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        return self.clf.fit(X, y)

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.clf.predict(X)

    def predict_proba(self, X):
        ''' Similar to predict, but probabilities of belonging to a class are output.'''
        return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes

    def get_classes(self):
        return self.clf.classes_

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self

if __name__=="__main__":
    # We can use this to run this file as a script and test the Preprocessor
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data"
        output_dir = "../results"
    else:
        input_dir = argv[1]
        output_dir = argv[2];

    basename = 'Iris'
    D = DataManager(basename, input_dir) # Load data
    print("*** Original data ***")
    print(D)

    Prepro = Preprocessor()

    # Preprocess on the data and load it back into D
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])
    D.feat_name = np.array(['PC1', 'PC2'])
    D.feat_type = np.array(['Numeric', 'Numeric'])

    # Here show something that proves that the preprocessing worked fine
    print("*** Transformed data ***")
    print(D)

    # Fonction de test
def test():
    # Load votre model
    mod = model()
    # 1 - créer un data X_random et y_random fictives: utiliser https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.rand.html
    X_random = numpy.random.rand(0.96,0.95,0.95,0.94,0.95,0.95,0.91,0.86,0.87,0.8)
    Y_random = numpy.random.rand(0.89,0.9,0.89,0.89,0.89,0.88,0.88,0.87,0.87,0.87)

    # 2 - Tester l'entrainement avec mod.fit(X_random, y_random)
    mod.fit(X_random,Y_random)
    # 3 - Test la prediction: mod.predict(X_random
    mod.predict(X_random)
    # Pour tester cette fonction *test*, il suffit de lancer la commande ```python sample_code_submission/model.py```
