# Projet Machine Learning — Prédiction du diabète

Projet end-to-end du module **Machine Learning / Deep Learning**, 

Classification binaire visant à prédire la présence ou l'absence de diabète à partir de mesures cliniques.

---

## Contexte

Le diabète est un enjeu majeur de santé publique. Ce projet applique une démarche ML complète — de l'analyse exploratoire à la comparaison de modèles — sur un cas de **classification médicale**.

**Question métier :** *Peut-on prédire le risque de diabète (Outcome = 1) à partir de variables cliniques (glycémie, IMC, âge, etc.) ?*

---

## Dataset

| Propriété | Détail |
|-----------|--------|
| **Nom** | Pima Indians Diabetes Database |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes) / [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| **Catégorie** | Source 3 (plateforme reconnue) — alignée sur la classification médicale (Stanford Data Sources) |
| **Taille** | 768 lignes, 8 features + 1 cible |
| **Cible** | `Outcome` (0 = non diabétique, 1 = diabétique) |

**Justification du choix :** dataset documenté, taille conforme au cahier des charges (> 500 lignes), problème clair, adapté à la comparaison d'algorithmes classiques et à la soutenance orale.

---

## Structure du dépôt

```text
ML/
├── data/
│   ├── diabetes.csv
│   └── README.md
├── notebooks/
│   ├── 01_eda_draft.ipynb
│   ├── 02_preprocessing_draft.ipynb
│   ├── 03_models_draft.ipynb
│   ├── final_project.ipynb
│   └── projet_ml_expert.ipynb   # notebook final autonome (Google Colab)
├── models/
│   └── best_model.joblib
├── presentation/
│   ├── soutenance.pptx
├── app/
│   └── streamlit_app.py
├── src/
│   ├── preprocessing.py
│   └── train.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone <url-du-depot>
cd ML
pip install -r requirements.txt
```

---

## Reproduction

**Notebook principal (Google Colab / local) :**

```bash
jupyter notebook notebooks/projet_ml_expert.ipynb
```

Sur Google Colab : importer `notebooks/projet_ml_expert.ipynb` — le dataset se télécharge automatiquement.

1. Ouvrir le notebook final :

```bash
jupyter notebook notebooks/final_project.ipynb
```

2. Entraîner les modèles en ligne de commande :

```bash
cd src
python train.py
```

3. Lancer l'application Streamlit (bonus) :

```bash
streamlit run app/streamlit_app.py
```

**Déploiement Streamlit Cloud :** connecter le dépôt GitHub sur [share.streamlit.io](https://share.streamlit.io) et pointer vers `app/streamlit_app.py`.

---

## Pipeline ML

1. **EDA** — distribution des classes, corrélations, valeurs aberrantes (zéros codés)
2. **Preprocessing** — imputation, standardisation, feature engineering, split stratifié 80/20
3. **Modélisation** — Logistic Regression, Random Forest, SVM, Gradient Boosting
4. **Évaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, validation croisée 5-fold
5. **Sélection** — meilleur modèle sauvegardé dans `models/best_model.joblib`

---

## Résultats principaux (jeu de test 20 %)

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 (mean) |
|--------|----------|-----------|--------|-----|---------|--------------|
| Logistic Regression | 0.72 | 0.62 | 0.54 | 0.57 | 0.83 | 0.64 |
| **Random Forest** | **0.75** | **0.64** | **0.63** | **0.64** | **0.83** | **0.65** |
| SVM | 0.75 | 0.67 | 0.56 | 0.61 | 0.81 | 0.60 |
| Gradient Boosting | 0.75 | 0.65 | 0.59 | 0.62 | 0.83 | 0.65 |

**Modèle retenu :** Random Forest (meilleur F1-score sur le test set).

---

## Soutenance

- Présentation : `presentation/soutenance.pptx` (8 slides, ~3 min)

---

## Auteur

**Emna** — Projet solo, module Machine Learning,

---

## Références

- [UCI Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes)
- [Stanford Data Sources](https://med.stanford.edu/sdsr/datasets.html)
- [Portail Open Data Tunisie](https://www.data.gov.tn/) (alternative locale)
