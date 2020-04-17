'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import neighbors
from sklearn.ensemble import VotingClassifier

from sklearn.pipeline import Pipeline
from preprocessing import Preprocessor

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
#score_dir = 'scoring_program/'
#from sys import path
#path.append(score_dir)
#from libscores import get_metric
#"metric_name, scoring_function = get_metric()
'''from zPreprocessor import Preprocessor'''

class model (BaseEstimator):


    '''
        __init__: constructeur de la classe model.
        arguments:
            self: classifier.
            classifier : le classifier(avec ses hyper-paramètres) du model à créer.
    '''
    def __init__(self,classifier=RandomForestClassifier(n_estimators=180, max_depth=None, max_features='auto')):
        print("CONSTRUCTEUR MODELE")
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        self.preprocess = PCA()
        self.clf =  Pipeline([
            ('preprocessing', self.preprocess),
            ('classification', RandomForestClassifier(n_estimators=180, max_depth=None, max_features='auto'))
        ])


    '''
        fit : méthode qui permet d'entraîner le modèle.
        arguments:
            self : le classifier à entraîner.
            X: l'ensemble de donnée d'entraînement.
            y: l'ensemble des étiquettes correspondant aux différentes classes des données de X.
    '''
    def fit(self, X, y):
        if self.clf is not None:
            print("fit en cours...")
            self.clf.fit(X,y)
        self.is_trained=True
        print("fit done")


    '''
        predict : méthode qui permet de prédire un ensemble d'étiquettes.
        arguments:
            self : le classifier à utiliser.
            X: l'ensemble de donnée.
        return:
            y : l'ensemble des étiquettes correspondant aux différentes classes des données de X.
    '''
    def predict(self, X):
        if self.clf is not None:
            print("predict en cours...")
            y = self.clf.predict(X)
        print("predict done")
        return y



    def save(self, path="./"):
        file = open(path + '_model.pickle', "wb")
        pickle.dump(self, file)
        file.close()



    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)

        return self


def test():
    model1 = model()
    X_random = np.random.rand(10000,203)
    Y_random = np.random.randint(1,7,size=(10000,1))
    Y_random = Y_random.ravel()
    X_train, X_valid, Y_train, Y_valid = train_test_split( X_random, Y_random, test_size=0.33, random_state=42)
    print(Y_train.shape)
    print(Y_valid.shape)
    model1.fit(X_train,Y_train)
    prediction = model1.predict(X_valid)
    #print(scoring_function(Y_valid,prediction))
    plt.scatter(np.arange(0,3300),Y_valid)
    plt.plot(np.arange(0,3300),prediction, c='r')

if __name__ == "__main__":
    data = load_wine()
    model1 = model()
    X = data.data
    Y = data.target
    Y = Y.ravel()
    X_train, X_valid, Y_train, Y_valid = train_test_split( X, Y, test_size=0.33, random_state=42)
    print(Y_train.shape)
    print(Y_valid.shape)
    model1.fit(X_train,Y_train)
    prediction = model1.predict(X_valid)
    #test()
