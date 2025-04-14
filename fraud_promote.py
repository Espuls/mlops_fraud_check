from mlflow.tracking import MlflowClient

# Constantes
TRACKING_URI = "sqlite:///mlflow.db"
F1_SCORE_METRIC = "f1_score"
STAGING_THRESHOLD = 0.56  # Promover ao Staging apenas acima desse valor

# Inicializar MlflowClient
client = MlflowClient(tracking_uri=TRACKING_URI)


def process_model_versions(model_name, versions):
    """
    Processa as versões de um modelo, promovendo ao Staging e identificando o melhor modelo para Produção.
    """
    best_model_version = None
    best_f1_score = 0

    for version in versions:
        run_id = version.run_id
        metrics = client.get_run(run_id).data.metrics

        if F1_SCORE_METRIC in metrics:
            f1_score = metrics[F1_SCORE_METRIC]

            # Promover ao Staging
            if f1_score > STAGING_THRESHOLD:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Staging"
                )
                print(f"Modelo {model_name} versão {version.version} com F1-score {f1_score} promovido para Staging.")

            # Identificar o melhor modelo para Produção
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_model_version = version.version

    # Promover o melhor modelo para Produção
    if best_model_version:
        client.transition_model_version_stage(
            name=model_name,
            version=best_model_version,
            stage="Production"
        )
        print(f"O modelo {model_name} versão {best_model_version} agora é o melhor com F1-score {best_f1_score}.")
    else:
        print(f"Nenhum modelo {model_name} atingiu os critérios de aceitação para ser considerado o melhor.")


# Processar os modelos RFC e DTC usando a função reutilizável
rfc_versions = client.search_model_versions("name='RandomForestClassifierSearch'")
process_model_versions("RandomForestClassifierSearch", rfc_versions)

dtc_versions = client.search_model_versions("name='DecisionTreeClassifierSearch'")
process_model_versions("DecisionTreeClassifierSearch", dtc_versions)