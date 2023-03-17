<<<<<<< HEAD
import pandas as pd

# ouvrir le fichier CSV dans un DataFrame
df = pd.read_csv('Lol_matchs.csv')

df.drop([0])


# afficher les 5 premières lignes du DataFrame
def trouver_colonnes_1(row):
    colonnes = []
    for nom_colonne, valeur in row.items():
        if valeur == 1:
            colonnes.append(nom_colonne)
    return colonnes


def trouver_colonnes_2(row):
    colonnes = []
    for nom_colonne, valeur in row.items():
        if valeur == 2:
            colonnes.append(nom_colonne)
    return colonnes


def trouver_colonnes_3(row):
    colonnes = []
    for nom_colonne, valeur in row.items():
        if valeur == 3:
            colonnes.append(nom_colonne)
    return colonnes


# appliquer la fonction à chaque ligne du DataFrame
df['colonnes_avec_1'] = df.apply(trouver_colonnes_1, axis=1)
df['colonnes_avec_2'] = df.apply(trouver_colonnes_2, axis=1)
df['colonnes_avec_3'] = df.apply(trouver_colonnes_3, axis=1)

lolo = df.apply(trouver_colonnes_1, axis=1)

print (lolo)


def concatener_lignes(row):
    return row['colonnes_avec_1'].str.cat([row['colonnes_avec_2'], row['colonnes_avec_3']], sep=' ')

# appliquer la fonction à chaque ligne du DataFrame
dernieres_colonnes = df.iloc[:, -3:]

# afficher le résultat
print(dernieres_colonnes)

matchs = []
# afficher le résultat
=======
import numpy as np
import pandas as pd
filename = "Lol_matchs.csv"
dataframe = pd.read_csv(filename,sep=";")
>>>>>>> origin/main
