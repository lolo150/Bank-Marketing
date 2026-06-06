# Bank Marketing — Prédiction de souscription à un dépôt à terme

Ce projet de Machine Learning analyse le dataset **Bank Marketing** afin de prédire si un client va souscrire ou non à un dépôt à terme.

L'objectif est de construire un pipeline simple et compréhensible allant de l'exploration des données jusqu'à l'entraînement et l'évaluation de modèles de classification.

## Objectif du projet

Le projet vise à répondre à la question suivante :

> À partir des informations client et des campagnes marketing précédentes, peut-on prédire si un client va souscrire à un dépôt à terme ?

La variable cible est `y` :

- `yes` : le client a souscrit au dépôt à terme ;
- `no` : le client n'a pas souscrit.

## Dataset utilisé

Le fichier utilisé est :

```text
bank-full.csv
```

Le dataset contient :

- **45 211 lignes**
- **17 colonnes**
- une variable cible : `y`
- des variables numériques et catégorielles

Le fichier CSV utilise le point-virgule comme séparateur. Il doit donc être chargé avec :

```python
pd.read_csv("bank-full.csv", sep=";")
```

## Variables du dataset

Le dataset contient notamment les variables suivantes :

| Variable | Description générale |
|---|---|
| `age` | âge du client |
| `job` | type de métier |
| `marital` | situation matrimoniale |
| `education` | niveau d'éducation |
| `default` | présence d'un défaut de paiement |
| `balance` | solde moyen annuel |
| `housing` | prêt immobilier |
| `loan` | prêt personnel |
| `contact` | type de contact utilisé |
| `day` | jour du dernier contact |
| `month` | mois du dernier contact |
| `duration` | durée du dernier contact |
| `campaign` | nombre de contacts pendant la campagne |
| `pdays` | nombre de jours depuis le dernier contact précédent |
| `previous` | nombre de contacts avant cette campagne |
| `poutcome` | résultat de la campagne précédente |
| `y` | variable cible |

## Structure du projet

```text
.
├── app_explique (1).ipynb
├── bank-full.csv
└── README.md
```

## Étapes réalisées dans le notebook

### 1. Importation des bibliothèques

Le projet utilise principalement :

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

### 2. Chargement et exploration des données

Le notebook commence par charger le dataset, afficher sa taille, ses colonnes, ses types de variables et vérifier la présence de valeurs manquantes classiques.

Le dataset ne contient pas de valeurs manquantes au format `NaN`, mais contient plusieurs valeurs écrites sous la forme `"unknown"`.

### 3. Analyse de la variable cible

La variable cible `y` est déséquilibrée :

| Classe | Nombre | Pourcentage |
|---|---:|---:|
| `no` | 39 922 | 88,30 % |
| `yes` | 5 289 | 11,70 % |

Ce déséquilibre est important, car le modèle pourrait facilement favoriser la classe majoritaire `no`.

### 4. Nettoyage des valeurs inconnues

Certaines valeurs `"unknown"` sont remplacées par des valeurs plus explicites :

- `job` → `no job`
- `education` → `no education`
- `contact` → valeur la plus fréquente hors `unknown`
- `poutcome` → `no previous campaign`

### 5. Préparation des données

La variable cible est transformée en variable numérique :

```text
no  → 0
yes → 1
```

Les données sont ensuite séparées en :

- variables explicatives `X`
- variable cible `y`

Puis un split train/test est réalisé avec :

- **80 %** pour l'entraînement
- **20 %** pour le test
- stratification sur la variable cible

### 6. Encodage des variables catégorielles

Les variables catégorielles sont encodées avec un **K-Fold Target Encoding**.

Cette méthode permet de transformer les catégories en valeurs numériques tout en limitant le risque de fuite de données entre le train et le test.

Variables catégorielles encodées :

```text
job, marital, education, default, housing, loan, contact, month, poutcome
```

### 7. Standardisation

Les données sont standardisées avec `StandardScaler`.

La standardisation est apprise uniquement sur le jeu d'entraînement, puis appliquée au jeu de test afin de respecter une bonne pratique de Machine Learning.

### 8. Modèles entraînés

Deux modèles sont entraînés et comparés :

#### Régression logistique

Paramètres principaux :

- `max_iter=3000`
- `class_weight="balanced"`
- `random_state=52`

#### Arbre de décision

Paramètres principaux :

- `max_depth=5`
- `class_weight="balanced"`
- `random_state=42`

## Résultats obtenus

### Régression logistique

| Métrique | Valeur |
|---|---:|
| Accuracy | 0.8424 |
| ROC-AUC | 0.9030 |
| Recall classe `yes` | 0.79 |
| F1-score classe `yes` | 0.54 |

### Arbre de décision

| Métrique | Valeur |
|---|---:|
| Accuracy | 0.7963 |
| ROC-AUC | 0.8848 |
| Recall classe `yes` | 0.83 |
| F1-score classe `yes` | 0.49 |

## Analyse des résultats

La régression logistique obtient la meilleure performance globale avec une ROC-AUC d'environ **0.90**.

L'arbre de décision obtient un rappel légèrement meilleur sur la classe `yes`, mais sa précision est plus faible. Cela signifie qu'il détecte davantage de clients susceptibles de souscrire, mais fait aussi plus d'erreurs.

Dans le contexte marketing, le recall de la classe `yes` est important, car l'objectif est souvent d'identifier un maximum de clients potentiellement intéressés.

## Variables importantes

Les variables les plus influentes dans les modèles sont notamment :

- `duration`
- `month`
- `poutcome`
- `housing`
- `pdays`

La variable `duration` apparaît comme particulièrement importante dans les deux modèles.

## Comment exécuter le projet

### 1. Cloner le dépôt

```bash
git clone https://github.com/lolo150/Bank-Marketing
cd Bank-Marketing
```

### 2. Installer les dépendances

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Lancer le notebook

```bash
jupyter notebook "app.ipynb"
```

### 4. Vérifier la présence du dataset

Le fichier suivant doit être placé dans le même dossier que le notebook :

```text
bank-full.csv
```

## Technologies utilisées

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Auteur

Projet réalisé par **lolo150**.

## Licence

Ce projet est destiné à un usage pédagogique et peut être adapté ou amélioré librement.
