from flask import Flask, request, jsonify
import mlflow, pandas as pd, logging, os, traceback
from mlflow.tracking import MlflowClient


###############################################################################
# CONFIG
###############################################################################
HOST, PORT = "127.0.0.1", 5000
MLFLOW_TRACKING_URI     = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME  = "mlops_credit_fraud"
MAX_CONTENT_LENGTH      = 10 * 1024 * 1024          # 10 MiB – raise if needed
EXPECTED_COLUMNS = (["Time"] +
    [f"V{i}" for i in range(1, 29)] +
    ["Amount"]
)                                                   # 31 features
###############################################################################

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def _latest_run_and_model_uri() -> str | None:
    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        logging.error("⚠  MLflow experiment %s not found", MLFLOW_EXPERIMENT_NAME)
        return None

    runs = mlflow.search_runs([exp.experiment_id],
                        order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        logging.error("⚠  No runs in experiment %s", MLFLOW_EXPERIMENT_NAME)
        return None

    run_id = runs.iloc[0].run_id
    cli = MlflowClient()
    for art in cli.list_artifacts(run_id):
        if art.is_dir and any(f.path.endswith("MLmodel")
                              for f in cli.list_artifacts(run_id, art.path)):
            return f"runs:/{run_id}/{art.path}"
    logging.error("⚠  Could not find a model directory in run %s", run_id)
    return None


def _load_model():
    uri = _latest_run_and_model_uri()
    if uri is None:
        return None

    try:
        logging.info("⏳ Loading model from %s …", uri)
        return mlflow.sklearn.load_model(uri)
    except Exception:                                 # fall back to generic
        logging.warning("🔄 Falling back to pyfunc loader")
        try:
            return mlflow.pyfunc.load_model(uri)
        except Exception as e:
            logging.error("❌  Failed to load model: %s\n%s", e,
                          traceback.format_exc())
            return None


MODEL = _load_model()
logging.info("✅ Model loaded: %s", "yes" if MODEL else "NO – health‑check will fail")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok" if MODEL else "error",
                    "model_loaded": bool(MODEL)}), (200 if MODEL else 503)


# accept both …/invocations and …/invocations/ so a stray slash never 404
@app.route("/invocations",  methods=["POST"])
@app.route("/invocations/", methods=["POST"])
def invocations():
    if MODEL is None:
        return jsonify(error="model_not_loaded"), 503

    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify(error="invalid_json"), 400

    if not isinstance(payload, dict) or "instances" not in payload:
        return jsonify(error="missing 'instances'"), 400

    df = pd.DataFrame(payload["instances"])
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        return jsonify(error=f"missing_columns: {missing}"), 400

    try:
        preds = MODEL.predict(df[EXPECTED_COLUMNS])
        return jsonify(predictions=preds.tolist())
    except Exception as e:
        logging.error("❌  Prediction error: %s\n%s", e, traceback.format_exc())
        return jsonify(error=str(e)), 500


if __name__ == "_main_":            # run WITHOUT the Werkzeug reloader
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)