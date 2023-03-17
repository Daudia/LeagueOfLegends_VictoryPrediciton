from sklearn.tree import DecisionTreeClassifier
import numpy as np
#Tableau X contenant les données d'entrainement et Y contenant les etiquettes des classes
X = np.array([ [1, 0, 0, 1, 1, 0, 0, 1, 0, 0], # première équipe
               [0, 1, 0, 0, 0, 1, 1, 0, 1, 0], # deuxième équipe
               ...
             ])
y = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0, ...]) #résultats des matchs

#Crée une instance de la classe DecisionTreeClassifier en spécifiant les paramètres de l'algorithme: le critère de séparation des nœuds (entropy) et la profondeur maximale de l'arbre.
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)

#On entraîne l'arbre de décision sur les données d'apprentissage
clf.fit(X, y)

