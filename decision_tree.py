from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
#Tableau X contenant les données d'entrainement et Y contenant les etiquettes des classes (70%)
X_train = np.array([ [1, 0, 0, 1, 1, 0, 0, 1, 0, 0], # première équipe
               [0, 1, 0, 0, 0, 1, 1, 0, 1, 0], # deuxième équipe
             ])
y_train = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0, ...]) #résultats des matchs

#Tableau X_test et Y_test contenant les données de test (30%)
X_test =np.array([ [1, 0, 0, 1, 1, 0, 0, 1, 0, 0], # première équipe
               [0, 1, 0, 0, 0, 1, 1, 0, 1, 0], # deuxième équipe
             ])
y_test = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0, ...]) #résultats des matchs
#DECISION TREE
#Crée une instance de la classe DecisionTreeClassifier en spécifiant les paramètres de l'algorithme: le critère de séparation des nœuds (entropy) et la profondeur maximale de l'arbre.
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)

#On entraîne l'arbre de décision sur les données d'apprentissage
clf.fit(X_train, y_train)
print(X_train.shape)

#On prédit les résultats des matchs sur les données de test
Y_pred = clf.predict(X_test)

#On évalue les performances du modèle sur les données d'entraînement et de test
train_score = clf.score(X_train, y_train)
print("Score de l'arbre de décision sur les données d'entraînement : {:.2f}".format(train_score))
test_score = clf.score(X_test, y_test)
print("Score de l'arbre de décision sur les données de test : {:.2f}".format(test_score))


#RANDOMFOREST

clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(X, Y)
prediction = clf.predict([[1, 1, 1, 1, 1] + [0 for  i in range(145)],
                          [1, 1, 1, 1, 0, 1] + [0 for  i in range(144)]])
print(prediction)
