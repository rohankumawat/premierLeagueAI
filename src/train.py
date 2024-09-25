import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from joblib import dump
import mlflow
import mlflow.sklearn
import os

# Function to load preprocessed data
def load_data(file_path):
    """
    Load preprocessed data from a CSV file.
    Args:
        file_path (str): Path to the preprocessed data file.
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# General training function with MLflow logging
def train_model(model, X_train, y_train, model_name):
    """
    Train a model with the provided training data and log with MLflow.
    Args:
        model: The machine learning model to train.
        X_train (pd.DataFrame): The input features for training.
        y_train (pd.Series): The target labels for training.
        model_name (str): Name of the model.
    Returns:
        Trained model.
    """
    try:
        # Ensure any active run is ended before starting a new one
        if mlflow.active_run() is not None:
            print(f"Ending existing active run with ID: {mlflow.active_run().info.run_id}")
            mlflow.end_run()

        # Start a new run
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(model, model_name)

        # Explicitly end the run after training
        mlflow.end_run()

        return model
    
    except Exception as e:
        print(f"Error in training {model_name}: {e}")
        
        # Ensure run is ended in case of an error
        if mlflow.active_run() is not None:
            mlflow.end_run()
        return None

# Model evaluation function with MLflow logging
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a model using test data and log metrics with MLflow.
    Args:
        model: The trained machine learning model.
        X_test (pd.DataFrame): The input features for testing.
        y_test (pd.Series): The target labels for testing.
        model_name (str): Name of the model.
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    preds = model.predict(X_test)
    precision = precision_score(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # Log metrics with MLflow
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)

    return {
        "precision": precision,
        "accuracy": accuracy,
        "roc_auc": roc_auc
    }

# Save trained model to disk
def save_model(model, model_name, model_dir="models"):
    """
    Save the trained model to disk using joblib.
    Args:
        model: The trained machine learning model.
        model_name (str): Name of the model.
        model_dir (str): Directory where the model will be saved.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    dump(model, model_path)
    print(f"Model saved to {model_path}")

# Main function
if __name__ == "__main__":
    # Paths to data
    preprocessed_data_path = "data/preprocessed/preprocessed_data.csv"
    
    # Load preprocessed data
    data = load_data(preprocessed_data_path)
    if data is None:
        print("Data loading failed. Exiting.")
        exit()

    # Features and target
    predictors = ["venue_code", "opp_code", "hour", "day_code", "gf_rolling", "ga_rolling", 
                  "sh_rolling", "sot_rolling", "dist_rolling", "fk_rolling", "pk_rolling", "pkatt_rolling"]
    target = "target"

    # Split the data into training and testing sets
    X = data[predictors]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
    }

    # Train and evaluate each model with MLflow
    for model_name, model in models.items():
        print(f"Training {model_name}...")

        # Train the model and log it to MLflow
        trained_model = train_model(model, X_train, y_train, model_name)

        if trained_model is not None:
            # Evaluate the model only if it was trained successfully
            metrics = evaluate_model(trained_model, X_test, y_test, model_name)
            print(f"{model_name} Evaluation:")
            print(f"Precision: {metrics['precision']}")
            print(f"Accuracy: {metrics['accuracy']}")
            print(f"ROC AUC: {metrics['roc_auc']}\n")
            
            # Save the trained model
            save_model(trained_model, model_name)
        else:
            print(f"Skipping evaluation for {model_name} due to training failure.")