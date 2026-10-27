# Credit Card Default Prediction

An educational historical-data classification pipeline with a FastAPI prediction interface. It fetches UCI dataset 350, preprocesses a train/test split, tunes three scikit-learn classifiers, and saves local artifacts. It is not a validated lending decision system.

## Pipeline

1. [loan_default_prediction.py](loan_default_prediction.py) fetches data through `ucimlrepo`, prints exploration output, and writes `loan_default_data.csv`.
2. [data_preprocessing.py](data_preprocessing.py) creates a stratified split, fits preprocessing on the training portion, applies SMOTE to that portion, and saves `preprocessor.joblib`.
3. [model_training.py](model_training.py) tunes **LogisticRegression, RandomForestClassifier, and GradientBoostingClassifier** with three-fold ROC-AUC grid search, prints held-out metrics, and saves each fitted model. It does not train XGBoost.
4. [app.py](app.py) loads `preprocessor.joblib` and `logisticregression_model.joblib` at import. Serving always selects Logistic Regression; it does not automatically deploy the highest-scoring model.

## Local setup and execution order

Use a fresh environment rather than the committed `venv`:

```bash
git clone https://github.com/Zinga18018/credit-card-default-prediction.git
cd credit-card-default-prediction
python -m venv .venv-local
source .venv-local/bin/activate
python -m pip install -r requirements.txt
python -m pip install imbalanced-learn fastapi uvicorn
```

On Windows, activate with `.venv-local\Scripts\activate`. The last command supplies imports missing from the unpinned [requirements.txt](requirements.txt); it is not a verified dependency lock.

After confirming the dataset's source and usage terms, the source-defined workflow is:

```bash
python loan_default_prediction.py
python data_preprocessing.py
python model_training.py
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The first step accesses the network and prints dataset rows. Training performs parallel grid searches and writes model files; it is not a lightweight startup check. Data acquisition and training also contain top-level work, so importing those modules is not side-effect-free. The required dataset and generated artifacts are not committed. The API exits during import if its artifacts are missing; load only trusted joblib files.

## API and evaluation limits

- `GET /health`: process health after artifact loading.
- `POST /predict/`: a binary prediction and model probability using the `X1`–`X23` schema in `app.py`.
- `/docs`: generated FastAPI documentation.

Prediction inputs are logged; do not submit sensitive customer records to this development service. Probabilities are not a guarantee of an individual's future behavior.

Preprocessing and SMOTE happen **before** GridSearchCV divides the training data into folds. This permits information sharing across validation folds and can make CV estimates optimistic. The held-out test split is separate, but this is not a leakage-free cross-validation pipeline. No committed evaluation aggregate establishes accuracy, calibration, fairness, or production suitability.

The acquisition source identifies UCI dataset 350; confirm its attribution and terms before redistribution. The earlier README labels the project MIT, but no project license file accompanies that statement. Dataset rights and code licensing are separate questions.
