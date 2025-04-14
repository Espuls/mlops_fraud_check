from mlflow.tracking import MlflowClient
from typing import List

# Constants
TRACKING_URI = "sqlite:///mlflow.db"


# Function to display models
def display_registered_models(registered_models: List):
    for model in registered_models:
        print(f"Model: {model.name}, Version: {model.latest_versions}")


# MLflow Client initialization
mlflow_client = MlflowClient(tracking_uri=TRACKING_URI)

# Search registered models and display them
registered_models = mlflow_client.search_registered_models()
display_registered_models(registered_models)