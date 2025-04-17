import pandas as pd
from monitor_dataset import preprocess_data
from models.decision_tree_class import train_decision_tree_model
from models.random_forest_class import train_random_forest_model
import mlflow


# Configurações globais
DATASET_PATH = "dataset/creditcard.csv"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "mlops_credit_fraud"


def setup_mlflow():
    """Configura o MLflow e cria o experimento se necessário."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        return experiment_id
    except Exception as e:
        print(f"Erro ao configurar MLflow: {e}")
        raise


def load_data(file_path):
    """Carrega o dataset a partir do caminho especificado."""
    return pd.read_csv(file_path)


def check_class_distribution(y):
    print("Distribuição das classes:")
    print(y.value_counts())
    print("\nProporção das classes:")
    print(y.value_counts(normalize=True))


def main():
    # Configura o MLflow primeiro
    experiment_id = setup_mlflow()
    print(f"MLflow configurado com experiment_id: {experiment_id}")

    # Carregando e pré-processando os dados
    print("Carregando e pré-processando dados...")
    data = load_data(DATASET_PATH)
    x, y = preprocess_data(data)

    # Verificar distribuição das classes
    check_class_distribution(y)

    # Treinamento e avaliação dos modelos
    print("\nTreinando Decision Tree...")
    decision_tree_model = train_decision_tree_model(x, y, data)
    if decision_tree_model is None:
        print("Falha no treinamento do Decision Tree")
        return

    print("\nTreinando Random Forest...")
    random_forest_model = train_random_forest_model(x, y, data)
    if random_forest_model is None:
        print("Falha no treinamento do Random Forest")
        return


if __name__ == "__main__":
    main()