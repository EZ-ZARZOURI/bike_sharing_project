# NOTE METHODE - Bike Sharing POV

## 1. Définition du problème et horizon de prédiction

- **Problème métier** : Prédire la demande horaire de vélos pour optimiser le rééquilibrage et la disponibilité.
- **Horizon de prédiction : Prévision horaire pour l’heure suivante (t+1) sur le test set.
- **Utilité pour un opérateur : Permet de planifier le rééquilibrage des stations à chaque heure, en s’assurant que les vélos sont disponibles là où la demande sera la plus forte.

---

## 2. Stratégie de validation

- Split temporel strict : 80% train / 20% test chronologique.
- Pas de shuffle → évite le **data leakage**.
- Test set = dernières heures du dataset (décembre 2025).

---

## 3. Baseline vs amélioration

| Modèle                        | MAE    | sMAPE   | Commentaire |
|--------------------------------|--------|---------|------------|
| Baseline Linear Regression      | 151.41 | 31.72% | Utilise uniquement les features calendaires (heure, jour, mois, weekend). Trop simple, ne capture pas les pics. |
| Random Forest                   | 73.13  | 14.72% | Utilise les lags 1h et 24h et rolling 24h. Capture mieux les variations et les pics. |

- La Random Forest divise par ~2 l’erreur moyenne.
- Les features lag/rolling sont cruciales pour la prédiction horaire.

---

## 4. Analyse de robustesse

- **Pics de demande** :
  - Nombre d’heures de pic dans le test set : 30
  - MAE sur pics : 1214.97 → très élevé, le modèle sous-estime les valeurs extrêmes.
- **Jours fériés** :
  - MAE sur jours fériés : 611.86 → erreurs importantes sur les jours atypiques.
- **Jours normaux** :
  - MAE sur jours normaux : 86.14 → proche de l’erreur globale, le modèle prédit bien les comportements standards.

💡 Interprétation :  
- Le modèle Random Forest fonctionne bien sur les jours classiques et capture les tendances moyennes.  
- Les erreurs extrêmes se produisent sur les **pics de demande** et les **jours atypiques** (fériés, événements rares).  
- Ces observations montrent la **limite de l’apprentissage sur des événements rares** et l’importance de features additionnelles pour améliorer la robustesse.

---

## 5. Limites du modèle

- **Pas encore pris en compte** :
  - Conditions météorologiques (pluie, tempêtes, températures extrêmes)
  - Jours fériés et événements spéciaux (manque de données pour les apprendre correctement)
  - Features station-level (le modèle est agrégé par heure)
- Certaines erreurs peuvent apparaître sur **pics extrêmes** ou jours atypiques.

---

## 6. Vision MLOps

Pour industrialiser ce modèle :

1. **Pipeline quotidien** :
   - Chargement automatique des nouveaux CSV
   - Mise à jour des features lag et rolling
   - Réentraînement ou ajustement du modèle
2. **Monitoring** :
   - Surveillance des MAE / sMAPE sur les nouvelles données
   - Alertes si l’erreur dépasse un seuil
3. **Réentraînement régulier** :
   - Une fois par mois ou après chaque nouveau batch de données
   - Versionnage des modèles avec Git / MLflow
4. **Exploitation** :
   - Intégration dans un tableau de bord pour l’opérateur
   - Export des prévisions horaires pour le planning logistique

---

## 7. Transparence

- **Temps passé** : ~4h  
- **Usage d’IA (ChatGPT)** : Aide pour structuration du code, debug et rédaction de README / NOTE_METHODE.
