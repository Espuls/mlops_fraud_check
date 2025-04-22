# MLOps Credit Fraud Detection

Este projeto implementa um sistema de detecção de fraudes em cartão de crédito usando técnicas de Machine Learning e MLOps.
- Trabalho final disciplina de MLOps
- Alunos: Erick Puls, Grace Feijó e Daniela Cavalheiro

## Configuração do Ambiente
1. Instale as dependências:
   * `pip install -r requirements.txt`

## Estrutura do Projeto

MLOps_credit_fraud/        
├── dataset/                
└── creditcard.csv         
├── models/                
├── decision_tree_class.py  
└── random_forest_class.py
├── .gitignore
├── app_server.py
├── dataset_check.py 
├── fraud_dataset.py
├── fraud_promote.py 
├── monitor_dataset.py 
├── README.md 
└── requirements.txt

## Ordem de Execução

1. **Verificação do Dataset**
   - Primeiro, certifique-se de que o arquivo `creditcard.csv` está presente na pasta `dataset/`
   - O dataset pode ser obtido em [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - A versão zipada do dataset está no repositório devido ao seu tamanho, é necessário fazer o unzip do arquivo

2. **Monitoramento dos Dados**
   * `python dataset_check.py`
   * `python monitor_dataset.py`
   - Estes scripts fazem o pré-processamento inicial dos dados
   - Geram estatísticas básicas sobre o dataset
   - Verificam a distribuição das classes

3. **Treinamento dos Modelos**
   * `python fraud_dataset.py`
   - Treina os modelos Decision Tree e Random Forest
   - Registra experimentos no MLflow
   - Gera métricas de avaliação

4. **Check / Promoção dos modelos registrados**
   * `python registry_model_check.py` e `fraud_promote.py`
   - Verifica os modelos registrados
   - Promove os modelos registrados que se destacaram em relação as métricas

5. **API endpoint**
   * `python app_server.py`
   - Configura o endpoint para a utilização do /invocations do mlflow
   - Contém as APIs aque podem ser utilzadas

## Mlflow UI

   - Para visualizar os experimentos no MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
   - Acesse `http://localhost:5000` no navegador

## Estrutura dos Arquivos Principais

1. **monitor_dataset.py / dataset_check.py**
   - Responsável pelo pré-processamento dos dados
   - Análise exploratória inicial
   - Verificação da qualidade dos dados

2. **fraud_dataset.py**
   - Script principal de treinamento
   - Configuração do MLflow
   - Execução dos modelos

3. **models/decision_tree_class.py**
   - Implementação do modelo Decision Tree
   - Tratamento de dados desbalanceados
   - Métricas de avaliação

4. **models/random_forest_class.py**
   - Implementação do modelo Random Forest
   - Otimização de hiperparâmetros
   - Métricas de avaliação

## Notas Importantes

   1. Dataset extremamente desbalanceado (0.17% de fraudes)
   2. Uso de técnicas de balanceamento (SMOTE)
   3. Métricas específicas para classes desbalanceadas
   4. Todos os experimentos são registrados no MLflow

## Monitoramento

   - Métricas são salvas no MLflow
   - Modelos são versionados
   - Experimentos podem ser comparados via UI do MLflow

## Troubleshooting

   - Se encontrar problemas com SMOTE ou balanceamento de classes:
   1. Verifique se há amostras suficientes da classe minoritária
   2. Ajuste os parâmetros de síntese de dados
   3. Verifique os logs do MLflow para detalhes dos erros

## Manutenção

   - Para atualizar as dependências:
   * `pip freeze > requirements.txt`
   * - Para limpar o banco do MLflow:
   * `del mlflow.db`
   - Para limpar os artefatos do MLflow:
   * `rd /s /q mlruns`
