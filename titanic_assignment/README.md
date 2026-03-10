# Titanic Dataset Analysis — Assignment 2

Predictive modelling for Titanic survival through data cleaning, feature
engineering and feature selection.

---

## Project Structure

```
titanic_assignment/
├── data/
│   ├── train.csv               ← raw training data (with Survived label)
│   ├── test.csv                ← raw test data (no label)
│   ├── train_cleaned.csv       ← output of data_cleaning.py
│   ├── train_engineered.csv    ← output of feature_engineering.py
│   ├── train_selected.csv      ← output of feature_selection.py
│   └── selected_features.txt  ← plain list of chosen features
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── feature_selection.py
├── README.md
└── requirements.txt
```

---

## How to Run

### 1 – Install dependencies

```bash
pip install -r requirements.txt
```

### 2 – Run the scripts in order

```bash
cd titanic_assignment
python scripts/data_cleaning.py        # → data/train_cleaned.csv
python scripts/feature_engineering.py # → data/train_engineered.csv
python scripts/feature_selection.py   # → data/train_selected.csv
```

### 3 – Explore the notebook

```bash
jupyter notebook notebooks/Titanic_Feature_Engineering.ipynb
```

---

## Approach

### Part 1 – Data Cleaning

| Column   | Issue                         | Strategy                                                |
|----------|-------------------------------|---------------------------------------------------------|
| Age      | ~20 % missing                 | Add binary indicator `HasAge`, then impute with median  |
| Embarked | 2 missing                     | Impute with mode (`S`)                                  |
| Fare     | Occasionally missing (test)   | Impute with median                                      |
| Cabin    | ~77 % missing                 | Extract first letter as `Deck`; unknown → `"Unknown"`  |
| Sex      | Potential casing issues       | Standardise to lowercase                                |
| Dupes    | —                             | Drop exact duplicate rows                               |

**Outlier handling:** IQR-based capping.  
- Fare: 3 × IQR (legitimate right skew — first-class fares are genuinely high).  
- Age: 1.5 × IQR (unusual ages beyond ~65 are capped).

---

### Part 2 – Feature Engineering

| Feature        | Description                                         |
|----------------|-----------------------------------------------------|
| FamilySize     | `SibSp + Parch + 1`                                 |
| IsAlone        | 1 if `FamilySize == 1`                              |
| Title          | Extracted from Name; rare titles grouped as "Rare"  |
| AgeGroup       | Child (<13) / Teen (13–17) / Adult (18–59) / Senior (60+) |
| FarePerPerson  | `Fare / FamilySize`                                 |
| LogFare        | `log1p(Fare)` — reduces right skew                 |
| LogAge         | `log1p(Age)` — mild skew reduction                 |
| Pclass_x_Fare  | Interaction: class × fare                           |
| Age_x_IsAlone  | Interaction: age × alone flag                       |

Categorical features (`Sex`, `Embarked`, `Title`, `Deck`, `AgeGroup`) are
one-hot encoded.  `Pclass` is kept as an ordinal integer (1 / 2 / 3).

---

### Part 3 – Feature Selection

1. **Correlation analysis** – highly correlated feature pairs (|r| > 0.90)
   are resolved by dropping the member with the lower absolute correlation
   with the target.
2. **Random Forest importance** – 200-tree classifier ranks all remaining
   features; top 15 retained.
3. **RFE (extra credit)** – Recursive Feature Elimination with Logistic
   Regression selects 15 features; unioned with the RF top-15 to form the
   final set.

---

## Key Observations

- **Title** is the single most predictive feature — "Mrs" and "Miss"
  strongly correlate with survival (women-first evacuation).
- **Pclass** and **Fare/LogFare** capture socioeconomic status; first-class
  passengers survived at much higher rates.
- **FamilySize / IsAlone** reveals that solo travellers and very large
  families had lower survival rates than small family groups.
- **Age** matters: children (especially in 2nd/3rd class) were prioritised
  in lifeboats.
- **Deck** (derived from Cabin) adds signal but has high missingness (~77 %),
  so "Unknown" is treated as a distinct category.
- **Fare** is right-skewed; log-transforming it improves model performance
  for distance-based algorithms.

