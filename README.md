4.  **Data Acquisition:**
    *   The dataset "Default of Credit Card Clients" will be automatically fetched using `ucimlrepo` when `loan_default_prediction.py` is run.

5.  **Run Initial Data Exploration:**
    ```bash
    python loan_default_prediction.py
    ```
    This script will fetch the dataset, perform initial data exploration, and save the raw data to `loan_default_data.csv`.

6.  **Data Preprocessing:**
    ```bash
    python data_preprocessing.py
    ```
    This script will load the raw data, perform data cleaning, feature engineering, split the data into training and testing sets, and save the preprocessor for later use.

7.  **Model Training and Evaluation:**
    ```bash
    python model_training.py
    ```
    This script will train various machine learning models, perform hyperparameter tuning using GridSearchCV, evaluate their performance, and save the best performing models.

8. **Model Deployment with FastAPI:**
    - Run the FastAPI application: `uvicorn app:app --reload`
    - Access the API documentation at `http://127.0.0.1:8000/docs`

9. **Testing the FastAPI Application:**
    - Open your web browser and navigate to `http://127.0.0.1:8000/docs`.
    - **Test the Health Endpoint:**
        - Expand the `/health` endpoint.
        - Click on "Try it out".
        - Click "Execute". You should receive a `200 OK` response with `{"status": "ok"}`.
    - **Test the Prediction Endpoint:**
        - Expand the `/predict/` endpoint.
        - Click on "Try it out".
        - In the "Request body" section, enter the following example input features:
        ```json
        {
          "credit_amount": 100000,
          "gender": 1,
          "education": 2,
          "marital_status": 1,
          "age": 30,
          "repayment_status_sep": -1,
          "repayment_status_aug": -1,
          "repayment_status_jul": -1,
          "repayment_status_jun": -1,
          "repayment_status_may": -1,
          "repayment_status_apr": -1,
          "bill_amount_sep": 50000,
          "bill_amount_aug": 45000,
          "bill_amount_jul": 40000,
          "bill_amount_jun": 35000,
          "bill_amount_may": 30000,
          "bill_amount_apr": 25000,
          "pay_amount_sep": 2000,
          "pay_amount_aug": 2000,
          "pay_amount_jul": 2000,
          "pay_amount_jun": 2000,
          "pay_amount_may": 2000,
          "pay_amount_apr": 2000
        }
        ```
        - Click "Execute". The response will include the `prediction` (0 for no default, 1 for default) and the `probability` of default, along with a `meaning` field.