import pandas as pd
import numpy as np
import requests
import os
from evidently.core.report import Report
from evidently.presets import DataDriftPreset, ClassificationPreset
from sklearn.preprocessing import LabelEncoder

# Constantes globais
API_URL = 'http://127.0.0.1:5000/invocations'
HEADERS = {'Content-Type': 'application/json'}
COLUMNS = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
    'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
    'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]


def detect_drift(drift_score, num_drift_columns):
    """Detecta drift e decide se é necessário treinar o modelo."""
    if drift_score > 0.5 or num_drift_columns > 2:
        print(f'Drift detectado com score {drift_score} e {num_drift_columns} colunas afetadas. Treinando modelo.')
        os.system('python fraud_dataset.py')
    else:
        print('Modelo está estável, sem necessidade de retreinamento.')


def preprocess_data(df, drop_columns=None, fill_value=0):
    """Pré-processa os dados removendo colunas, ajustando tipos e preenchendo valores ausentes."""
    if drop_columns:
        df.drop(columns=drop_columns, inplace=True, errors='ignore')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.infer_objects(copy=False)
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('float64')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])
    df.fillna(fill_value, inplace=True)

    x = df.drop(columns=['Class'])
    y = df['Class']
    print('Pré-processamento dos dados concluído.')
    return x, y.astype(int)


def load_new_data(file_path='dataset/creditcard.csv'):
    """Carrega novos dados do dataset e o pré-processa."""
    df = pd.read_csv(file_path).sample(frac=1).reset_index(drop=True)
    return preprocess_data(df)


def simulate_drift(df_examples):
    """Simula drift gerando alterações nos dados."""
    new_df = df_examples.copy()
    new_df['Time'] = np.random.randint(0, 100, len(new_df))
    new_df['Amount'] = np.random.randint(0, 1000, len(new_df))
    print('Dataset artificial criado para simular drift.')
    return new_df


def get_predictions(data):
    """Obtém previsões do modelo via API."""
    instances = [{col: row[col] for col in COLUMNS} for _, row in data.iterrows()]
    response = requests.post(API_URL, headers=HEADERS, json={'instances': instances})
    print(response.status_code)
    print(response.json())
    predictions = response.json().get('predictions', [])
    print(predictions)
    return predictions


def evaluate_model_drift(reference_data, y, current_data=None):
    """Avalia o drift do modelo comparando os dados de referência com os dados atuais."""
    target_df = reference_data.copy() if current_data is None else current_data.copy()
    missing_cols = [col for col in COLUMNS if col not in reference_data.columns]
    if missing_cols:
        print(f"❌ Columns missing in data: {missing_cols}")
        return []
    target_df["prediction"] = get_predictions(target_df)
    target_df["target"] = y

    report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
    report.run(reference_data=reference_data, current_data=target_df)
    report.save_html("monitoring_report.html")

    result = report.to_dict()
    drift_score = result["metrics"][0]["result"]["dataset_drift"]
    drift_by_columns = result["metrics"][1]["result"].get("drift_by_columns", {})
    print(f"Drift score: {drift_score}")
    print(f"Drift by columns: {drift_by_columns}")
    return drift_score, drift_by_columns


def main():
    df_examples, y = load_new_data()
    drift_score, drift_by_columns = evaluate_model_drift(df_examples, y)
    new_data = simulate_drift(df_examples)
    drift_score, drift_by_columns = evaluate_model_drift(df_examples, y, new_data)
    detect_drift(drift_score, drift_by_columns)


if __name__ == '__main__':
    main()