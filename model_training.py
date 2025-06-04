import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib

# Load preprocessed data
try:
    df = pd.read_csv('loan_default_data.csv')
    preprocessor = joblib.load('preprocessor.joblib')
    # Import preprocess_data function from data_preprocessing.py
    from data_preprocessing import preprocess_data
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)
except FileNotFoundError:
    print("Error: 'loan_default_data.csv' or 'preprocessor.joblib' not found. Please run previous steps.")
    exit()

# Define models and their hyperparameters for GridSearchCV
models = {
    'LogisticRegression': {
        'model': LogisticRegression(solver='liblinear', random_state=42),
        'params': {
            'C': [0.1, 1.0, 10.0]
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        }
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1]
        }
    }
}

best_models = {}

print("\n--- Model Training and Hyperparameter Tuning ---")
for name, config in models.items():
    print(f"\nTraining {name}...")
    grid_search = GridSearchCV(config['model'], config['params'], cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_

    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best ROC AUC score for {name}: {grid_search.best_score_:.4f}")

print("\n--- Model Evaluation on Test Set ---")
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n{name} Evaluation:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")

    # Save the best performing model
    joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.joblib')
    print(f"Best {name} model saved to {name.lower().replace(' ', '_')}_model.joblib")