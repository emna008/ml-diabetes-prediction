# Préparation soutenance orale — Questions fondamentales ML

Document de préparation pour la séance Q&R (4 minutes, 3 questions).

---

## Question 1 : Pourquoi utiliser F1-score et Recall plutôt que l'Accuracy seule ?

**Réponse :**

- Le dataset est **déséquilibré** : environ 65 % de patients non diabétiques et 35 % diabétiques.
- L'**accuracy** peut être trompeuse : un modèle qui prédit toujours « pas de diabète » obtiendrait ~65 % d'accuracy sans être utile cliniquement.
- Le **Recall (rappel)** mesure la proportion de vrais diabétiques correctement détectés. En santé, un **faux négatif** (diabète non détecté) est coûteux.
- Le **F1-score** combine precision et recall en une seule métrique, utile quand on cherche un compromis entre détecter les cas positifs et limiter les fausses alertes.

**Lien avec notre projet :** Random Forest a été retenu car il maximise le F1 sur le jeu de test tout en maintenant un ROC-AUC élevé (~0.83).

---

## Question 2 : Comment fonctionne Random Forest ?

**Réponse :**

- Random Forest est un ensemble (**bagging**) de **arbres de décision** entraînés en parallèle.
- Chaque arbre apprend sur un **échantillon bootstrap** des données et un **sous-ensemble aléatoire de features** à chaque split.
- La prédiction finale est le **vote majoritaire** (classification) de tous les arbres.
- Cette randomisation **réduit le sur-apprentissage** par rapport à un arbre unique.
- On peut interpréter le modèle via les **feature importances** (ex. Glucose, BMI).

**Lien avec notre projet :** Random Forest capture les relations non linéaires entre Glucose, BMI et la cible, ce que la régression logistique modélise moins bien.

---

## Question 3 : Qu'est-ce que la validation croisée et le sur-apprentissage ?

**Réponse :**

- Le **sur-apprentissage (overfitting)** survient quand le modèle mémorise le train set et généralise mal sur de nouvelles données (écart train/test élevé).
- La **validation croisée k-fold** (k=5 ici) divise les données en k parties : le modèle est entraîné k fois sur k-1 folds et évalué sur le fold restant.
- La moyenne des scores CV donne une estimation **plus robuste** de la performance que un seul split.
- Pour limiter le sur-apprentissage : régularisation (Logistic Regression), profondeur limitée des arbres, GridSearchCV, et pipeline sans fuite de données.

**Lien avec notre projet :** Nous avons utilisé `cross_val_score` avec scoring F1 et un split 80/20 stratifié. Le preprocessing est dans un Pipeline sklearn pour que le scaler ne voie jamais les données de test pendant l'entraînement.

---

## Questions bonus à anticiper

### Différence entre Régression Logistique et SVM

- **Logistic Regression** : modèle linéaire probabiliste, interprétable (coefficients).
- **SVM** : cherche une frontière de décision maximisant la marge entre classes ; kernel RBF pour relations non linéaires. Nécessite le scaling.

### Rôle du StandardScaler

- SVM et la régression logistique sont sensibles à l'échelle des variables.
- StandardScaler centre et réduit les features (moyenne 0, écart-type 1) pour que chaque variable contribue équitablement.

### Interprétation de la matrice de confusion

|  | Prédit 0 | Prédit 1 |
|--|----------|----------|
| **Réel 0** | Vrais négatifs (TN) | Faux positifs (FP) |
| **Réel 1** | Faux négatifs (FN) | Vrais positifs (TP) |

- **Precision** = TP / (TP + FP) — fiabilité des alertes positives.
- **Recall** = TP / (TP + FN) — capacité à détecter les diabétiques.

---

## Timing soutenance (3 min slides)

| Slide | Durée approx. |
|-------|---------------|
| 1 Titre | 15 s |
| 2 Dataset | 25 s |
| 3 EDA | 25 s |
| 4 Preprocessing | 25 s |
| 5 Modèles | 20 s |
| 6 Résultats | 30 s |
| 7 Modèle retenu | 20 s |
| 8 Conclusion | 20 s |
| **Total** | **~3 min** |
