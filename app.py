from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import logging
import os

app = FastAPI()

# Load the preprocessor and the best model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('logisticregression_model.joblib') # Assuming Logistic Regression is chosen as the best model
    logger.info("Model and preprocessor loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"Error loading model or preprocessor: {e}. Please run model_training.py first.")
    exit()
except Exception as e:
    logger.error(f"An unexpected error occurred during model/preprocessor loading: {e}")
    exit()

class PredictionInput(BaseModel):
    credit_amount: int = Field(..., alias="X1", description="Amount of the given credit (NT dollar)")
    gender: int = Field(..., alias="X2", description="Gender (1 = male; 2 = female)")
    education: int = Field(..., alias="X3", description="Education (1 = graduate school; 2 = university; 3 = high school; 4 = others)")
    marital_status: int = Field(..., alias="X4", description="Marital status (1 = married; 2 = single; 3 = others)")
    age: int = Field(..., alias="X5", description="Age (year)")
    repayment_status_sep: int = Field(..., alias="X6", description="Repayment status in September, 2005 (-1=pay duly, 1=delay 1 month, ..., 9=delay 9+ months)")
    repayment_status_aug: int = Field(..., alias="X7", description="Repayment status in August, 2005")
    repayment_status_jul: int = Field(..., alias="X8", description="Repayment status in July, 2005")
    repayment_status_jun: int = Field(..., alias="X9", description="Repayment status in June, 2005")
    repayment_status_may: int = Field(..., alias="X10", description="Repayment status in May, 2005")
    repayment_status_apr: int = Field(..., alias="X11", description="Repayment status in April, 2005")
    bill_amount_sep: int = Field(..., alias="X12", description="Amount of bill statement in September, 2005 (NT dollar)")
    bill_amount_aug: int = Field(..., alias="X13", description="Amount of bill statement in August, 2005")
    bill_amount_jul: int = Field(..., alias="X14", description="Amount of bill statement in July, 2005")
    bill_amount_jun: int = Field(..., alias="X15", description="Amount of bill statement in June, 2005")
    bill_amount_may: int = Field(..., alias="X16", description="Amount of bill statement in May, 2005")
    bill_amount_apr: int = Field(..., alias="X17", description="Amount of bill statement in April, 2005")
    pay_amount_sep: int = Field(..., alias="X18", description="Amount of previous payment in September, 2005 (NT dollar)")
    pay_amount_aug: int = Field(..., alias="X19", description="Amount of previous payment in August, 2005")
    pay_amount_jul: int = Field(..., alias="X20", description="Amount of previous payment in July, 2005")
    pay_amount_jun: int = Field(..., alias="X21", description="Amount of previous payment in June, 2005")
    pay_amount_may: int = Field(..., alias="X22", description="Amount of previous payment in May, 2005")
    pay_amount_apr: int = Field(..., alias="X23", description="Amount of previous payment in April, 2005")

@app.post("/predict/")
async def predict(input_data: PredictionInput):
    logging.info(f"Received prediction request with data: {input_data.dict()}")

    # Convert input data to a pandas DataFrame
    input_df = pd.DataFrame([input_data.dict(by_alias=True)])

    # Ensure the columns are in the expected order
    expected_columns = [
        "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11",
        "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21",
        "X22", "X23"
    ]
    input_df = input_df.reindex(columns=expected_columns)

    logging.info(f"DataFrame columns after reindexing: {input_df.columns.tolist()}")
    logging.info(f"DataFrame shape after reindexing: {input_df.shape}")

    # Preprocess the input data
    try:
        processed_data = preprocessor.transform(input_df)
    except ValueError as e:
        logging.error(f"ValueError during preprocessing: {e}")
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")

    # Make prediction
    prediction_proba = model.predict_proba(processed_data)[:, 1]
    prediction = (prediction_proba >= 0.5).astype(int)

    prediction_meaning = "The model predicts that the client WILL default on their credit card payment." if prediction[0] == 1 \
                         else "The model predicts that the client will NOT default on their credit card payment."

    return {"prediction": int(prediction[0]), "probability": float(prediction_proba[0]), "meaning": prediction_meaning}

@app.get("/health")
async def health_check():
    return {"status": "ok"}