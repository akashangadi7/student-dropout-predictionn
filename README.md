# Student Dropout and Academic Success Prediction

Machine learning project that predicts whether a university student will
**Drop out**, stay **Enrolled**, or **Graduate**, and explains each prediction with SHAP.
Made for PRG4206.

**Author:** Akash Angadi, I26044162

## Dataset
"Predict Students' Dropout and Academic Success" (4,424 students, 34 features + target).
Source: Realinho, Machado, Baptista and Martins, "Predicting Student Dropout and
Academic Success", Data, vol. 7, no. 11, 2022 (doi: 10.3390/data7110146).
The file is in `data/dataset.csv`.

## Models
- Logistic Regression (baseline)
- Random Forest
- XGBoost

The data is split 70/15/15 (train/validation/test). Explainability uses SHAP.

## Files
| File | What it does |
|---|---|
| `train_models.py` | EDA, trains the 3 models, evaluation, SHAP, saves figures and the best model |
| `early_warning.py` | Tests prediction using enrollment data only, then + 1st semester, then + 2nd semester |
| `app.py` | Streamlit demo app |
| `results.csv`, `results_early_warning.csv` | Result tables |
| `figures/` | Charts used in the report |

## Install
Python 3.12 was used.
```
pip install -r requirements.txt
```

## How to run
Run in this order, from the project folder:
```
python train_models.py
python early_warning.py
streamlit run app.py
```
`train_models.py` creates `models/best_model.joblib`, which the app needs, so run it first.
The `models` folder is not in this repository because the file is too large.

## Results (test set)
| Model | Accuracy |
|---|---|
| Logistic Regression | 76.2% |
| Random Forest | 79.8% |
| XGBoost | 80.0% |

Early-warning test (XGBoost): about 65% accuracy with enrollment data only,
about 77% after adding 1st-semester results, and about 80% with all data.
The Enrolled class is the hardest to predict.

## Limitations
Data comes from one Portuguese university and one train/test split was used.
This is an educational prototype. It should not be used to make decisions about students.
