import pandas as pd
import mlflow
from monitor_dataset import preprocess_data
from models.decision_tree_class import train_decision_tree_model
from models.random_forest_class import train_random_forest_model

# Configurações globais
DATASET_PATH = "dataset/creditcard.csv"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "mlops_credit_fraud"


def setup_mlflow():
    """Configura o MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def load_data(file_path):
    """Carrega o dataset a partir do caminho especificado."""
    return pd.read_csv(file_path)


def main():
    setup_mlflow()

    # Carregando e pré-processando os dados
    data = load_data(DATASET_PATH)
    x, y = preprocess_data(data)

    # Treinamento dos modelos
    decision_tree_model = train_decision_tree_model(x, y, data)
    random_forest_model = train_random_forest_model(x, y, data)


if __name__ == "__main__":
    main()