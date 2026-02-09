# Bike Sharing POV - Capital Bikeshare

## 1. Contexte

Ce projet vise à transformer les données brutes de Capital Bikeshare (2024-2025) en un **Proof of Value exploitable** pour un opérateur de mobilité.  
L'objectif est de prédire la demande horaire de vélos afin d'optimiser le rééquilibrage et la disponibilité.

---

## 2. Structure du projet

bike_sharing_project/
├── data/
│ └── raw/ # Contient les fichiers ZIP avec les CSV de Capital Bikeshare
├── src/
│ ├── train.py # Script principal pour entraîner et tester les modèles
│ ├── data_loader.py # Chargement et agrégation des données
│ ├── features.py # Création des features temporelles et des lags
│ ├── model.py # Modèles baseline et Random Forest
│ └── evaluate.py # Fonctions pour calculer MAE, sMAPE
├── requirements.txt # Dépendances Python
├── README.md # Ce fichier
└── NOTE_METHODE.md # Explication méthodologique

---

## 3. Installation

1. Cloner le dépôt ou dézipper le projet.
2. Installer les dépendances Python : pip install -r requirements.txt

Vérifier que les fichiers ZIP sont dans data/raw/ :

data/raw/202401-capitalbikeshare-tripdata.zip
data/raw/202402-capitalbikeshare-tripdata.zip

---

## 4. Exécution

Depuis le dossier racine :

cd src; python train.py

Le script effectuera :

Chargement et agrégation des données horaires

Création des features temporelles et lags

Split temporel train/test (80% / 20%)

Entraînement :

Baseline Linear Regression

Random Forest avec lags et rolling

Évaluation sur le test set : MAE et sMAPE

. Performances obtenues
Modèle	MAE	sMAPE
Baseline Linear Regression	151.41	31.72%
Random Forest	73.13	14.72%

Le Random Forest montre une nette amélioration grâce aux features lag et rolling.

---

## 5. Analyse de robustesse (Random Forest)
Catégorie	MAE	Commentaire
Pics de demande	1214.97	Modèle sous-estime fortement les valeurs extrêmes
Jours fériés	611.86	Erreurs importantes sur les jours atypiques
Jours normaux	86.14	Erreur proche de l’erreur globale, modèle fiable sur les comportements standards

💡 Interprétation :

Le modèle est performant sur la majorité des heures normales.

Les erreurs extrêmes apparaissent sur les événements rares (pics et jours fériés).
