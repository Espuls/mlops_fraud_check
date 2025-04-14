import pandas as pd
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def log_metrics(metrics):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def evaluate_model(model_info, x_test, y_test, feature_columns):
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = loaded_model.predict(x_test)
    result = pd.DataFrame(x_test, columns=feature_columns)
    result["label"] = y_test.values
    result["predictions"] = predictions
    mlflow.evaluate(
        data=result,
        targets="label",
        predictions="predictions",
        model_type="classifier",
    )
    print(result[:5])


def train_decision_tree_model(features, labels, dataset):
    model_name = "DecisionTreeClassifier"
    log_model_name = "DecisionTreeClassifierSearch"

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    with mlflow.start_run():
        mlflow.log_param("model_type", model_name)

        model = DecisionTreeClassifier(random_state=42)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        metrics = calculate_metrics(y_test, y_pred)
        log_metrics(metrics)

        mlflow.set_tag("dataset_used", dataset)
        signature = infer_signature(x_train, y_pred)

        model_info = mlflow.sklearn.log_model(
            model, model_name,
            signature=signature,
            input_example=x_train,
            registered_model_name=log_model_name
        )

        evaluate_model(model_info, x_test, y_test, features.columns.values)

        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
        return model