# Credit Card Default Prediction

A machine learning pipeline to predict credit card default, exposed as a REST API via FastAPI.

## Features

- **Automated data acquisition** from the UCI ML Repository using `ucimlrepo`
- **Preprocessing pipeline** with feature engineering, scaling, and train/test splitting
- **Multi-model training** with GridSearchCV hyperparameter tuning (XGBoost, Random Forest, Logistic Regression)
- **FastAPI deployment** with health check and prediction endpoints
- **Swagger UI** for interactive API testing at `/docs`

## Quick Start

```bash
git clone https://github.com/Zinga18018/credit-card-default-prediction.git
cd credit-card-default-prediction
pip install -r requirements.txt

# Step 1: Fetch data and explore
python loan_default_prediction.py

# Step 2: Preprocess
python data_preprocessing.py

# Step 3: Train models
python model_training.py

# Step 4: Serve API
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` to interact with the API.

## Architecture

```
credit-card-default-prediction/
  loan_default_prediction.py   # Data acquisition and exploration
  data_preprocessing.py        # Cleaning, feature engineering, splitting
  model_training.py            # Training, tuning, evaluation
  app.py                       # FastAPI REST API
  requirements.txt             # Dependencies
```

## API Endpoints

| Method | Endpoint    | Description                    |
|--------|-------------|--------------------------------|
| GET    | `/health`   | Server health check            |
| POST   | `/predict/` | Predict default probability    |

## Tech Stack

Python, FastAPI, scikit-learn, XGBoost, Pandas, NumPy, ucimlrepo

## License

MIT
