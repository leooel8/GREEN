'''
    Classe 'model' du groupe GREEN

        Circé CARLETTI et Léo RESSAYRE
        Dernière modification: 17/04

'''

'''
    Imports utiles au bon déroulement du programme
'''
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
score_dir = 'scoring_program/'
problem_dir = 'ingestion_program/'
from sys import path
path.append(score_dir)
path.append(problem_dir)
from libscores import get_metric
metric_name, scoring_function = get_metric()
from preprocessing import Preprocessor

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

'''
    Imports utiles à la classe modèle
'''
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from modelComparer import modelComparer

'''
    Imports utiles aux ensembles de données
'''
from sklearn.datasets import load_wine
import pandas as pd
from data_io import read_as_df
from data_manager import DataManager
data_dir = 'public_data'
data_name = 'plankton'

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
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        self.preprocess =  Preprocessor()
        #self.clf = classifier
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

'''
    Fonction de test utile au choix du modèle et de ses paramètres
'''
def test():
    # Définition de l'ensemnle de données et affichages
    print("=============================================================")
    print("=============================================================")
    print("                      Test de model.py                       ")
    print("             et comparaison avec d'autres modèles            ")
    print(" ")
    print(" ")
    print("             Chargement des données..")
    print(" ")
    data = read_as_df(data_dir  + '/' + data_name)
    print(data['target'].value_counts())
    print(" ")
    print("_____________________________________________________________")
    print(" ")
    print(data.head())
    print(" ")
    D = DataManager(data_name, data_dir, replace_missing=True)
    print("_____________________________________________________________")
    print(" ")
    print(D)
    print(" ")
    print("_____________________________________________________________")
    print(" ")
    print(" ")
    X_t = D.data['X_train']
    Y = D.data['Y_train']
    Y = Y.ravel()
    print("_____________________________________________________________")
    print(" ")
    print("     Division des données en deux ensembles (training et validation)")
    X_train, X_valid, Y_train, Y_valid = train_test_split( X_t, Y, test_size=0.33, random_state=42)
    print("Dimensions de Y_train")
    print(Y_train.shape)
    print("Dimensions de Y_valid")
    print(Y_valid.shape)
    print("DONE")
    print(" ")
    print(" ")
    print("_____________________________________________________________")
    print(" Comparaison des modèles : ")
    print(" ")
    print(" ")
    model1 = model()
    modelTest = modelComparer("Pipeline RandomForestClassifier avec Preprocessor()", model1)
    modelTest.addClassifier("OneVsOneClassifier",OneVsOneClassifier(SGDClassifier(random_state=42)))
    modelTest.addClassifier("AdaBoostClassifier",AdaBoostClassifier(n_estimators=100))
    modelTest.addClassifier("RandomForestClassifier",RandomForestClassifier(n_estimators=180, max_depth=None, max_features='auto'))
    modelTest.addClassifier("KNeighborsClassifier",neighbors.KNeighborsClassifier(n_neighbors=7))

    #Fonction 'fit' pour tous les modèles
    modelTest.fitAll(X_train,Y_train)

    modelTest.comparingFunction(X_train,Y_train,X_valid,Y_valid,5)

if __name__ == "__main__":
    # data = load_wine()
    # model1 = model()
    # X_random = data.data
    # Y_random = data.target
    # Y_random = Y_random.ravel()
    # X_train, X_valid, Y_train, Y_valid = train_test_split( X_random, Y_random, test_size=0.33, random_state=42)
    # print(Y_train.shape)
    # print(Y_valid.shape)
    # model1.fit(X_train,Y_train)
    # prediction = model1.predict(X_valid)
    test()
