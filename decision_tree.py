from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import time
from random import randint

start_time = time.time()

# Ouvrir le fichier CSV dans un DataFrame
df = pd.read_csv('final_dataset.csv')
df = df.drop(columns="num_match")
df1 = df.drop(columns="Team1_victory")
df2 = df.iloc[:, -1:]

X = df1.to_numpy()
Y = df2.to_numpy().ravel() # ravel converts column ndarray to row

print("> Opening and storing data took %.3f seconds" % (time.time() - start_time))

def random_input():
    randoms1 = [randint(0,149) for x in range(5)]
    randoms2 = [randint(0,149) for x in range(5)]
    print([df.columns.tolist()[randoms1[i]] for i in range(5)])
    print([df.columns.tolist()[randoms2[i]] for i in range(5)])
    X = [0 for x in range(300)]
    for r in randoms1:
        X[r] = 1
    for r in randoms2:
        X[r + 150] = 1
    return np.array(X)


start_time = time.time()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
clf = RandomForestClassifier(n_estimators=100)
# clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, max_features=150)
clf = clf.fit(X_train, Y_train)
nb_success = 0
Y_pred = clf.predict(X_test)
nb_test = len(Y_pred)
for i in range(len(Y_pred)):
    if(Y_pred[i] == Y_test[i]):
        nb_success += 1
print("Score :" + str(nb_success/nb_test))
print("Score : " + str(clf.score(X_test, Y_test)))

print("> Fitting model took %.3f seconds" % (time.time() - start_time))

print("\nSame teams prediction :")
prediction = clf.predict_proba([[1,1,1,1,1] + [0 for  i in range(145)] + [1,1,1,1,1] + [0 for  i in range(145)]])
print(prediction)

print("\nDifferent teams prediction : ")

#for i in range(10):
 #   print(clf.predict_proba([random_input()]))
  #  print()


name_list = df.columns.tolist()
t1 = ['Janna','Soraka','Nami','Blitzcrank','Lulu']
t2 = ['Garen','Syndra','Shaco','Caitlyn','Thresh']
x1 = []
x2 = []
for val in t1:
    x1.append(name_list.index(val))
for val in t2:
    x2.append(name_list.index(val))

X = [0 for x in range(300)]
for val in x1:
    X[val] = 1
for val in x2:
    X[val + 150] = 1


print(clf.predict_proba([X]))
print(clf.predict([X]))