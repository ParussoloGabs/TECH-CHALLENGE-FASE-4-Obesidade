# Tech Challenge Fase 4 - Análise Clínica de OBESIDADE

## Predicao do nivel de obesidade com machine learning

Este projeto foi estruturado para atender ao desafio da Fase 4 com foco em:

- treinamento de um modelo de classificacao multiclasse
- explicacao clara da pipeline de machine learning
- suporte para deploy com Streamlit

O dataset esperado e o `Obesity.csv`, cuja coluna alvo e `Obesity`.

## Estrutura dos arquivos

- `README.md`: guia rapido de uso do projeto
- `train_model.py`: script de treinamento, avaliacao e salvamento do pipeline
- `app_streamlit.py`: aplicacao Streamlit para realizar previsoes

## O que o treinamento faz

O script `train_model.py` executa as seguintes etapas:

1. carrega o dataset CSV
2. remove duplicatas
3. cria a feature `BMI`
4. cria versoes arredondadas de variaveis ordinais
5. separa treino e teste com `stratify`
6. compara dois modelos candidatos:
   - `LogisticRegression`
   - `RandomForestClassifier`
7. escolhe o melhor modelo com base na media de acuracia em validacao cruzada
8. avalia o melhor pipeline no conjunto de teste
9. salva o pipeline treinado e um arquivo de metricas

## Requisitos (Caso utilize ambiente local)

### Criar e ativar ambiente virtual (Windows PowerShell)

Abra o PowerShell na pasta do projeto e execute:

```powershell
# Se receber erro de execução de scripts, rode uma vez como admin:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
.\venv\Scripts\Activate.ps1

# Confirme que o venv está ativo: o terminal exibirá "(venv)" no prompt
```

### Instalar dependências

Com o venv ativo:

```powershell
pip install --upgrade pip
pip install pandas scikit-learn joblib streamlit
```

## Como treinar o modelo

Certifique-se de que o venv está ativo (veja `(venv)` no prompt do PowerShell).

Exemplo usando o CSV na mesma pasta do projeto:

```powershell
python train_model.py --data Obesity.csv
```

Exemplo informando o caminho completo do CSV:

```powershell
python train_model.py --data "Passe o nome do arquivo o caminho do arquivo. Ex: C:/path/exemplo.csv"
```

## Arquivos gerados no treinamento

O script cria uma pasta `Artefatos` com:

- `obesity_pipeline.joblib`: pipeline treinado
- `metrics.json`: metricas do experimento

## Como executar o Streamlit

Com o venv ativo:

```powershell
streamlit run app_streamlit.py
```

O app abrirá no navegador em `http://localhost:8501`. Se o modelo estiver salvo no local padrão `artifacts/obesity_pipeline.joblib`, a aplicacao o carrega automaticamente.

## Como subir a aplicação (Deploy da Solução)

O projeto já está preparado para deploy com Streamlit. Antes de publicar, confirme que estes arquivos estão presentes no repositório:

- `requirements.txt`
- `app_streamlit.py`
- `artifacts/obesity_pipeline.joblib`
- `artifacts/metrics.json`
- `artifacts/predictions.csv`

### Publicação no Streamlit Community Cloud

1. faça o push do projeto para um repositório GitHub
2. conecte o repositório ao Streamlit Community Cloud
3. use `app_streamlit.py` como arquivo principal
4. deixe o ambiente instalar as dependências a partir do `requirements.txt`
5. mantenha a pasta `Artefatos` versionada no repositório para que o app carregue o modelo e as análises

Se preferir re-treinar os artefatos antes de publicar, rode novamente:

```powershell
python train_model.py --data "Seu arquivo (Ex. Obesity.csv)"
```

Depois disso, os arquivos em `Artefatos` já ficam prontos para a aplicação subir com a página de predição, a página de analytics e a página técnica.

## Como testar o resultado

O fluxo recomendado para validar o projeto e este:

1. executar o treino com o CSV base
2. conferir se a pasta `Artefatos` foi criada com `obesity_pipeline.joblib` e `metrics.json`
3. abrir o Streamlit com `streamlit run app_streamlit.py`
4. confirmar na tela principal os resultados do treino, incluindo melhor modelo, acuracia holdout, comparacao dos candidatos, classification report e matriz de confusao
5. preencher os dados de um paciente e clicar em `Realizar previsao`

Se quiser um teste rapido, use os valores padrao da tela e apenas clique em `Realizar previsao`. Se quiser uma validacao mais realista, copie uma linha do seu CSV de treino, ajuste os campos para o formulario e compare a classe prevista com a classe real.

Com a atualizacao do app, a interface mostra duas partes principais:

- um resumo do treinamento com as metricas gravadas em `artefatos/metrics.json`
- o formulario de entrada para gerar a previsao em tempo real

