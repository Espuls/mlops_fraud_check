import pandas as pd, numpy as np, requests, os, logging, time
from evidently.core.report import Report
from evidently.presets import DataDriftPreset, ClassificationPreset
from sklearn.preprocessing import LabelEncoder


# Constantes globais
API_URL   = "http://127.0.0.1:5000/invocations"     # keep without trailing slash
HEADERS   = {"Content-Type": "application/json"}
BATCHSIZE = 512

COLUMNS = (["Time"] +
    [f"V{i}" for i in range(1, 29)] +
    ["Amount"]
)                                                   # 31 features

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


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


# --------------------------------------------------------------------------- #
#  Helper: POST in batches with retry / health‑check
# --------------------------------------------------------------------------- #
def _post_instances(instances, max_retries=3, backoff=2):
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(API_URL, headers=HEADERS, json={"instances": instances},
                              timeout=30)
            if r.status_code == 200:
                return r.json()["predictions"]
            raise RuntimeError(f"API {API_URL} -> {r.status_code} : {r.text[:200]}")
        except Exception as e:
            if attempt == max_retries:
                raise
            logging.warning("⚠  POST failed (%s) – retrying in %ss …", e, backoff)
            time.sleep(backoff)
    # unreachable


def get_predictions(df: pd.DataFrame) -> list[int]:
    """Returns predictions for df in order, batching to keep payloads small."""
    preds = []
    for start in range(0, len(df), BATCHSIZE):
        batch = df.iloc[start:start + BATCHSIZE][COLUMNS]
        instances = batch.to_dict(orient="records")
        preds.extend(_post_instances(instances))
    if len(preds) != len(df):
        raise RuntimeError(f"Prediction length mismatch ({len(preds)} vs {len(df)})")
    return preds


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