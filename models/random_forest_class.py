import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
import mlflow
from mlflow.models.signature import infer_signature


def calculate_metrics(y_true, y_pred):
    """Calcula métricas com tratamento para classes desbalanceadas"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)
    }


def train_random_forest_model(features, labels, dataset):
    model_name = "RandomForestClassifier"

    # Converte para numpy arrays
    x = features.to_numpy()
    y = labels.to_numpy()

    # Identifica os índices das classes
    fraud_indices = np.where(y == 1)[0]
    non_fraud_indices = np.where(y == 0)[0]

    print(f"\nTotal de amostras de fraude: {len(fraud_indices)}")

    # Cria as amostras sintéticas mais diversas
    n_synthetic = 100
    synthetic_samples = []
    synthetic_labels = []

    for idx in fraud_indices:
        fraud_sample = x[idx]
        for _ in range(n_synthetic // len(fraud_indices)):
            noise = np.random.normal(0, 0.1, size=fraud_sample.shape)
            synthetic_sample = fraud_sample + noise
            synthetic_samples.append(synthetic_sample)
            synthetic_labels.append(1)

    # Adiciona as amostras sintéticas
    if synthetic_samples:
        synthetic_x = np.vstack(synthetic_samples)
        synthetic_y = np.array(synthetic_labels)
        x = np.vstack([x, synthetic_x])
        y = np.append(y, synthetic_y)

    print(f"Amostras de fraude após síntese: {len(np.where(y == 1)[0])}")

    # Dividi os dados mantendo proporção
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_idx, test_idx in sss.split(x, y):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    print("\nDistribuição das classes:")
    print("Treino:", np.bincount(y_train.astype(int)))
    print("Teste:", np.bincount(y_test.astype(int)))

    try:
        # Aplica o SMOTE com as amostras sintéticas
        smote = SMOTE(
            random_state=42,
            k_neighbors=min(5, len(np.where(y_train == 1)[0]) - 1),
            sampling_strategy={1: min(sum(y_train == 0), 1000)}
        )

        x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

        print("\nDistribuição após balanceamento:")
        print("Treino balanceado:", np.bincount(y_train_balanced.astype(int)))

        # Configura e treina o Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(x_train_balanced, y_train_balanced)

        # Predições com threshold ajustado
        y_pred_proba = model.predict_proba(x_test)[:, 1]
        threshold = 0.3
        y_pred = (y_pred_proba > threshold).astype(int)

        print("\nDistribuição das predições:")
        print("Predições:", np.bincount(y_pred.astype(int)))

        metrics = calculate_metrics(y_test, y_pred)
        print("\nMétricas do modelo:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        # Prepara o exemplo de input para MLflow
        input_example = pd.DataFrame(x_train[:5], columns=features.columns)

        # Loga no MLflow com assinatura
        signature = infer_signature(x_train, y_pred)

        with mlflow.start_run(run_name=model_name) as run:
            # Loga os parâmetros
            mlflow.log_params({
                "model_type": model_name,
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "threshold": threshold,
                "class_weight": "balanced"
            })

            # Loga as métricas
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            # Loga o dataset e o modelo
            mlflow.set_tag("dataset_used", dataset)
            model_info = mlflow.sklearn.log_model(
                model,
                model_name,
                signature=signature,
                input_example=input_example
            )

            run_id = run.info.run_id

        print(f"\nModelo salvo em run {run_id}")
        return model

    except Exception as e:
        print(f"Erro durante o treinamento: {str(e)}")
        print(f"Detalhes adicionais:\n{type(e).__name__}: {str(e)}")
        return None