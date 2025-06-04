import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    # Separate target variable
    X = df.drop('Y', axis=1)
    y = df['Y']

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply preprocessing to the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Apply SMOTE to the training data to handle imbalance
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    return X_train_resampled, X_test_processed, y_train_resampled, y_test, preprocessor

if __name__ == "__main__":
    try:
        df = pd.read_csv('loan_default_data.csv')
    except FileNotFoundError:
        print("Error: 'loan_default_data.csv' not found. Please run loan_default_prediction.py first.")
        exit()

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    print("\n--- Data Preprocessing Complete ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Save preprocessor for later use in prediction pipeline
    import joblib
    joblib.dump(preprocessor, 'preprocessor.joblib')
    print("Preprocessor saved to preprocessor.joblib")