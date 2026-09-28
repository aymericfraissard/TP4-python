import pandas as pandasLib  
import random

dataFrame1MonJeuDeDonneeDentrainement = pandasLib.read_csv("donnevrai.csv")

print(dataFrame1MonJeuDeDonneeDentrainement)

dataFrame1MonJeuDeDonneeDentrainement["danger"] = dataFrame1MonJeuDeDonneeDentrainement["danger"].astype(int)

X = dataFrame1MonJeuDeDonneeDentrainement["vitesse"].tolist() 
Y = dataFrame1MonJeuDeDonneeDentrainement["danger"].tolist()

print(X[:5], Y[:5])



w = random.randint(-10, 10)
b = random.randint(-10, 10)

print(w, b)

def Zpredict(x):
    z = w * x + b
    if z >= 0:
        return 1
    else:
        return 0

for x in X:
    print(f"x = {x}, prédiction = {Zpredict(x)}")

for x, y in zip(X, Y):
    y_pred = Zpredict(x)
    erreur = y - y_pred
    print(f"x = {x}, y_attendu = {y}, y_predit = {y_pred}, erreur = {erreur}")