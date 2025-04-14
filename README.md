# mlops_fraud_check
trabalho final disciplina de mlops
* A versão zipada do dataset está no repositório devido ao seu tamanho, é necessário fazer o unzip do arquivo.
* Através desde link, a página do Kaggle do dataset pode ser acessada: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data.
* O arquivo dataset_check.py pode ser executado para a inspeção de dados e visualização da distribuição de classes do dataset.
* É necessário rodar o seguinte comando no terminal da IDE para executar o mlflow localmente: 'mlflow ui --backend-store-uri sqlite:///mlflow.d'.
* Executar fraud_dataset.py para setup do mlflow, carregamento e pré'processamento do dataset e treinamento dos modelos.
* Executar monitor_dataset.py para verificar drift e fazer a avaliação dos modelos.
* Executar registry_model_check.py para iniciar o client mlflow e mostrar os modelos registrados.
* Executar o fraud_promote.py para promover os modelos para produção