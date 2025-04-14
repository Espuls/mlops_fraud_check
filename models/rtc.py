import pandas as pd
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

# Constants for testing and model reproducibility
TEST_SIZE = 0.2
RANDOM_STATE = 42


def compute_metrics(y_test, y_pred):
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def log_mlflow_metrics(metrics):
    """Log metrics into MLflow."""
    for name, value in metrics.items():
        mlflow.log_metric(name, value)


def log_mlflow_model(model, x_train, y_pred, registered_name):
    """Log the MLflow model with signature and input example."""
    signature = infer_signature(x_train, y_pred)
    return mlflow.sklearn.log_model(
        model,
        "RandomForestClassifier",
        signature=signature,
        input_example=x_train,
        registered_model_name=registered_name,
    )


def train_random_forest_model(x, y, dataset):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForestClassifier")

        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Compute and log metrics
        metrics = compute_metrics(y_test, y_pred)
        log_mlflow_metrics(metrics)

        # Log the dataset as a tag and save model
        mlflow.set_tag("dataset_used", dataset.name)
        model_info = log_mlflow_model(model, x_train, y_pred, "RandomForestClassifierSearch")

        # Evaluate the model
        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        predictions = loaded_model.predict(x_test)
        result = pd.DataFrame(x_test, columns=x.columns.values)
        result["label"] = y_test.values
        result["predictions"] = predictions
        mlflow.evaluate(
            data=result,
            targets="label",
            predictions="predictions",
            model_type="classifier",
        )

        # Display results
        print(result[:5])
        mlflow.sklearn.log_model(model, "random_forest_classifier_model")
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
        return model