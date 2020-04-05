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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from libscores import get_metric
metric_name, scoring_function = get_metric()

''' 
   Classe "modelComparison"
   
       L'intérêt de cette classe est de pouvoir utiliser plusieurs modèles facilement, et de pouvoir les comparer de manière simple et efficace. On a accès aux fonctions de base d'un modèle (fit, predict) mais aussi de Cross-validation et de ValidationCurve pour pouvoir améliorer et valider un modèle.
       
   
'''    
class modelComparer(BaseEstimator):
    '''
        Constructeur
            @param classifier: un modèle
            @param name: le nom que l'on veut donner à notre modèle
            
            Crée un tableau composé d'un seul modèle
    '''
    def __init__(self, name,classifier):
        self.clfTab = [[name,classifier]]
     
    '''
        Fonction 'addClassifier'
            @param name: le nom que l'on veut donner au modèle à ajouter
            @param classifier: le modèle à ajouter
            
            Ajoute à l'instance un modèle dans le tableau des modèles 'clfTab'
    '''
    def addClassifier(self,name,classifier):
        self.clfTab.append([name,classifier])
        print("'",name,"' ajouté..")
    
    '''
        Fonction 'getClassifier'
            @param clfNum: l'indice du classifier que l'on veut récupérer
            
            Renvoie le classifier du tableau de classifier dont l'indice est passé en paramètre
    '''
    def getClassifier(self, clfNum):
        return self.clfTab[clfNum - 1][1]
    '''
        Fonction 'showClasses'
        
            Affiche les modèles actuellement présents dans le tableau des modèles 'clfTab'
    '''
    def showClasses(self):
        for i in range (0,len(self.clfTab)):
            print("Classifier", i + 1, ":", self.clfTab[i][0], ";")
    
    '''
        Fonction 'fitOne'
            @param clfNum: l'indice du modèle pour lequel on veut appliquer la fontion 'fit'
            @param X: l'ensemble des données sur lesquelles le modèle sera entrainé
            @param y: l'ensemble des labels sur lesquels le modèle sera entrainé
            
            Applique la fonction 'fit' au modèle dont l'indice est passé en paramètre
    '''
    def fitOne(self,clfNum,X,y):
        print("Classifier", clfNum , ":", self.clfTab[clfNum - 1][0], " fontion 'fit' en cours..")
        self.clfTab[clfNum - 1][1].fit(X,y)
        print("DONE")
        return self
    
    '''
        Fonction 'fitAll'
            @param X: l'ensemble des données sur lesquelles les modèles seront entrainés
            @param y: l'ensemble des labels sur lesquels les modèles seront entrainés
            
            Applique la fonction 'fit' à tous les modèles
    '''
    def fitAll(self,X,y):
        for i in range (0,len(self.clfTab)):
            print("Classifier", i + 1, ":", self.clfTab[i][0], " fontion 'fit' en cours..")
            self.clfTab[i][1].fit(X,y)
        print("DONE")
        return self
    
    '''
        Fonction 'predictOne'
            @param clfNum: l'indice du modèle pour lequel on veut appliquer la fonction 'predict'
            @param X: l'ensemble des données sur lesquelles le modèle effectuera une prédiction
            
            Applique la fonction 'predict' au modèle dont l'indice est passé en paramètre
    '''
    def predictOne(self,clfNum,X):
        return self.clfTab[clfNum - 1][1].predict(X)
    
    '''
        Fonction 'predictAll'
            @param X: l'ensemble des données sur lesquelles les modèles effectueront une prédiction
            @param y: l'ensemble des labels sur lesquels les modèles effectueront une prédiction
            
            Applique la fonction 'predict' à tous les modèles
    '''
    def predictAll(self,X,Y):
        for i in range (0,len(self.clfTab)):
            Y_hat = self.clfTab[i][1].predict(X)
            print("Classifier", i + 1, ":", self.clfTab[i][0], "PredictionScore:",scoring_function(Y,Y_hat))
    
    '''
        Fontion 'predict_proba'
            @param clfNum: l'indice du modèle sur lequel appliquer la fonctioin 'predict_proba'
            @param X: l'ensemble des données sur lesquelles appliquer la fonction 'predict_proba'
            
            Applique la fonction 'predict_proba' au modèle dont l'indice est passé en paramètre
    '''
    def predict_proba(self,clfNum,X):
        return self.clfTab[clfNum - 1][1].predict_proba(X)
    
    
    '''
        showPrediction
                @param clfNum: l'indice du classifier dont on veut afficher les résultats
                @param X: l'ensemble des données sur lesquelles les modèles effectueront une prédiction
                @param y: l'ensemble des labels sur lesquels les modèles effectueront une prédiction 
                
                Affiche un graphe pour visualiser le fonctionnement des prédictions du modèle
                
                WARNING 04/04/20: NE FONCTIONNE PAS
    '''
    def showPrediction (self, clfNum, X,Y):
        prediction = self.clfTab[clfNum - 1][1].predict(X)
        print("=================================================")
        print("=================================================")
        print(" Affichage des prédictions par le modèle")
        plt.scatter(X,Y)
        plt.plot(X,prediction,c='r',lw=3)
        
        
    '''
        Fonction 'crossValidationAll'
            @param X: l'ensemble des données sur lesquelles sera basé la cross-validation
            @param Y: l'ensemble des labels sur lesquels sera basé la cross-validation
            @param crossValNum: le nombre de découpage différent des données dans la cross-validation
            
            Effectue la cross-validation de tous les modèles sur les données passées en paramètre
    '''
    def crossValidationAll(self,X,Y,crossValNum):
        for i in range (0,len(self.clfTab)):
            crossValScore=cross_val_score(self.clfTab[i][1],X,Y,cv=crossValNum)
            print("Classifier", i + 1, ":", self.clfTab[i][0], "Cross Validation Score:",crossValScore.mean())
    
    '''
        Fonction 'comparingFunction'
            @param X_train: ensemble des données d'entrainement
            @param Y_train: ensemble des labels d'entrainement
            @param X_valid: ensemble des données de validation
            @param Y_valid: ensemble des labels de validation
            @param crossValNum: le nombre de découpage différent des données lors des cross-validations
            
            Cette fonction propose après son exécution, un affichage des scores de validation, ainsi que des scores de cross-validation pour chacuns des classifiers présents dans l'instance. De plus, la fonction indique à la suite les classifiers le plus performant pour les parties Validation et Cross-validation 
    '''
    def comparingFunction(self,X_train,Y_train,X_valid,Y_valid,crossValNum):
        for i in range (0,len(self.clfTab)):
            currentY_hat = self.clfTab[i][1].predict(X_valid) 
            scorePredict = scoring_function(Y_valid,currentY_hat)
            currentCvVal = cross_val_score(self.clfTab[i][1],X_train,Y_train,cv=crossValNum).mean()
            if i == 0:
                allTab = [[self.clfTab[0][0],scorePredict,currentCvVal]]
            if i != 0:
                allTab.append([self.clfTab[i][0],scorePredict,currentCvVal])
        bestScoreIndice = 0
        bestCrossIndice = 0
        print("===================================================")
        print("===================================================")
        print("                 Start of Comparing                ")
        print(" ")
        for u in range (0,len(allTab)):
            print("------------------------------------------------------")
            if allTab[u][1] > allTab[bestScoreIndice][1]:
                bestScoreIndice = u
            if allTab[u][2] > allTab[bestCrossIndice][2]:
                bestCrossIndice = u
            print("Classifier", u + 1, ":",allTab[u][0], " : Validation Score ->", allTab[u][1])
            print("                                     Cross Validation Score ->", allTab[u][2])
            print("------------------------------------------------------")
        print(" ")
        print(" ")
        print("--------------------- Results ---------------------")
        print("Best classifier for Validation Score : ", allTab[bestScoreIndice][0], "with the score : ", allTab[bestScoreIndice][1])
        print("Best classifier for Cross Validation score : ", allTab[bestCrossIndice][0], "with the score : ", allTab[bestCrossIndice][2])
        print(" ")
        print(" ")
        print("                  End of Comparing                 ")
        print("===================================================")
        print("===================================================")
        
        
        '''
            Fonction 'showValidationCurve'
                @param clfNum: indice du modèle pour lequel on veut utiliser la fonction de ValidationCurve
                @param X_train: ensemble des données sur lesquelles sera utilisé la fonction de ValidationCurve
                @param Y_train: ensemnle des labels sur lesquels sera utilisé la fonction de ValidationCurve
                @param paramName: le nom du paramètre que l'on veut modifier
                @param borneInf: la valeur de la borne inferieur de l'intervalle dans lequel devront se trouver les différentes valeurs que prendra le parmètre
                @param borneSup: la valeur de la borne superieur de l'intervalle dans lequel devront se trouver les différentes valeurs que prendra le parmètre
                @param crossValNum: la fonction ValidationCurve utilisant la Cross-Validation, ce paramètre indiquera le nombre de différents découpage qui sera fait dans la Cross-Validation
                
                Cette fonction affiche un graphe indiquant l'évolution des scores de validation et d'entrainement selon la valeur du paramètre modifié
        '''
    def showValidationCurve(self,clfNum,X_train,Y_train,
                            paramName,borneInf,borneSup,crossValNum):
        k = np.arange(borneInf,borneSup)
        train_score, val_score = validation_curve(self.clfTab[clfNum - 1][1],X_train,Y_train,paramName,k,cv=crossValNum)
        
        print("=================================================================")
        print("=================================================================")
        print(" ")
        print("     Evolution des scores en fonction de '",paramName,"'")
        print(" ")
        print(" ")
        plt.plot(k,val_score.mean(axis=1), label= 'validation')
        plt.plot(k, train_score.mean(axis=1), label='train')
        plt.ylabel('score')
        plt.xlabel(paramName)
        plt.legend()
        print(" ")
        print(" ")
        print("Infos du graphe: Classifier",clfNum,": ",self.clfTab[clfNum - 1][0])
        print("                 ",paramName," from ",borneInf," to ",borneSup)
        print("                 Cross Validation Number : ", crossValNum)
        print(" ")
        print(" ")
        print("=================================================================")
        print("=================================================================")
    
    '''
        Fonction showGridSearchCV
    '''
    def showGridSearchCV(self, clfNum, X_train, Y_train,X_valid,Y_valid, paramGrid, crossValNum):
        grid = GridSearchCV(self.clfTab[clfNum - 1][1], paramGrid, cv = crossValNum)
        grid.fit(X_train,Y_train)
        print("=================================================================")
        print("=================================================================")
        print("                Recherche des meilleurs paramètres               ")
        print("   Classifier utilisé: Classifier ", clfNum, " : ", self.clfTab[clfNum - 1][0])
        print(" ----------------------------RESULTATS---------------------------")
        print(" -> Meilleur score d'apprentissage obtenue: ", grid.best_score_, ";")
        print(" -> Paramètres correspondants: ", grid.best_params_, ";")
        temp = grid.best_estimator_
        print(" -> Score de Validation obtenu avec ces paramètres: ", temp.score(X_valid, Y_valid))
        print("=================================================================")
        print("=================================================================")
    '''
        Les fontions 'save' et 'load' sont les mêmes foncitons que pour les autres versions de 'model.py' à l'exception du fait qu'un paramètre 'clfNum' leur a été ajouté. Ce paramètre permet de choisir lequel des modèles présents dans l'instance sera utilisé dans les deux fonctions
    '''
    def save(self, clfNum, path="./"):
        file = open(path + '_model.pickle', "wb")
        pickle.dump(self.clfTab[clfNum][1], file)
        file.close()
    
    def load(self, clfNum, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self.clfTab[clfNum][1] = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        
        return self.clfTab[clfNum][1]
