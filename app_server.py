from flask import Flask, request, jsonify
import mlflow
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Configurar MLflow
os.environ['MLFLOW_TRACKING_URI'] = "sqlite:///mlflow.db"
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def load_latest_model():
    try:
        # Buscar o experimento
        experiment = mlflow.get_experiment_by_name("mlops_credit_fraud")
        if experiment is None:
            print("Experimento 'mlops_credit_fraud' não encontrado")
            return None

        # Buscar a última run
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )

        if len(runs) == 0:
            print("Nenhuma run encontrada")
            return None

        run_id = runs.iloc[0].run_id
        print(f"Carregando modelo da run: {run_id}")

        # Tentar carregar o modelo como sklearn
        try:
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            print("Modelo carregado com sucesso!")
            return model
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            try:
                # Tentar carregar como modelo genérico
                model = mlflow.pyfunc.load_model(model_uri)
                print("Modelo carregado como pyfunc!")
                return model
            except Exception as e:
                print(f"Erro ao carregar modelo como pyfunc: {e}")
                return None

    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None


# Carregar o modelo no início
print("Inicializando servidor...")
MODEL = load_latest_model()
print(f"Modelo carregado: {'Sim' if MODEL is not None else 'Não'}")


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'model_type': str(type(MODEL)) if MODEL is not None else None
    })


@app.route('/invocations', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Modelo não carregado'}), 500

    try:
        data = request.get_json()
        print("Dados recebidos:", data)  # Debug

        if not data or 'instances' not in data:
            return jsonify({'error': 'Dados inválidos'}), 400

        # Converter para DataFrame
        df = pd.DataFrame(data['instances'])
        print(f"Shape dos dados: {df.shape}")  # Debug

        # Fazer predições
        if isinstance(MODEL, mlflow.pyfunc.PyFuncModel):
            predictions = MODEL.predict(df)
        else:
            predictions = MODEL.predict(df)

        predictions = predictions.tolist() if isinstance(predictions, np.ndarray) else list(predictions)

        return jsonify({'predictions': predictions})

    except Exception as e:
        print(f"Erro durante a predição: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)