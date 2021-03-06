'''
    Classe 'model' du groupe GREEN

        Circé CARLETTI et Léo RESSAYRE
        Dernière modification: 17/04

'''

'''
    Imports utiles au bon déroulement du programme
'''
import sys
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sys import path
from preprocessing import Preprocessor
from sklearn.ensemble import VotingClassifier

'''
    Imports utiles pour les tests de modèles
'''
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import neighbors
from sklearn import ensemble

'''
    Imports utiles à la classe modèle
'''
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

'''
    Imports utiles aux ensembles de données
'''


'''
    Notre classe 'model'
        Doit contenir: fit
                       predict
                       save
                       load
'''
class model (BaseEstimator):
    '''
        __init__: constructeur de la classe model.
        arguments:
            self: classifier.
            classifier : le classifier(avec ses hyper-paramètres) du model à créer (RandomForestClassifier par défault).
    '''
    def __init__(self,classifier=RandomForestClassifier(n_estimators=180, max_depth=None, max_features='auto')):
        print("CONSTRUCTEUR MODELE")
        self.classifierUsed = classifier

        self.preprocess =  Preprocessor()
        #self.clf = classifier
        PipelineUse =  Pipeline([
            ('preprocessing', self.preprocess),
            ('classification',self.classifierUsed)
        ])

        self.clf = VotingClassifier(estimators=[
					('Gradient Tree Boosting', ensemble.GradientBoostingClassifier()),
					('Pipeline', PipelineUse),
                    ('RandomForestClassifier',RandomForestClassifier(n_estimators=180, max_depth=None, max_features='auto'))],
					voting='soft')

        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
    '''
        fit : méthode qui permet d'entraîner le modèle.
        arguments:
            self : le classifier à entraîner.
            X: l'ensemble de donnée d'entraînement.
            y: l'ensemble des étiquettes correspondant aux différentes classes des données de X.
    '''
    def fit(self, X, y):
        print("FIT")
        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1]
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")

        if self.clf is not None:
            self.clf.fit(X,y)

        self.is_trained=True
        print("FIT DONE")

    '''
        predict : méthode qui permet de prédire un ensemble d'étiquettes.
        arguments:
            self : le classifier à utiliser.
            X: l'ensemble de donnée.
        return:
            y : l'ensemble des étiquettes correspondant aux différentes classes des données de X.
    '''
    def predict(self, X):
        print("PREDICT")
        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        y = np.zeros([num_test_samples, self.num_labels])
        if self.clf is not None:
            y = self.clf.predict(X)
        print("PREDICT DONE")
        return y

    '''
        Fonction get_classes
    '''
    def get_classes(self):
        return self.clf.classes_

    '''
        Fonction save
    '''
    def save(self, path="./"):
        file = open(path + '_model.pickle', "wb")
        pickle.dump(self, file)
        file.close()

    '''
        Fonction load
    '''
    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)

        return self
    '''
        FIN CLASSE 'model'
    '''
