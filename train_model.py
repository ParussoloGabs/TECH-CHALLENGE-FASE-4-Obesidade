from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# APONTAMENTO DAS COLUNAS E VARIÁVEL TARGET.
# Tambem contém a categorizacao das colunas em numericas, categoricas e ordinais, para facilitar o processo de engenharia de features e pré-processamento.
TARGET_COLUMN = "Obesity"
ORDINAL_COLUMNS = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]
RAW_NUMERIC_COLUMNS = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
RAW_CATEGORICAL_COLUMNS = [
    "Gender",
    "family_history",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]
ENGINEERED_NUMERIC_COLUMNS = [f"{column}_rounded" for column in ORDINAL_COLUMNS]


# A Engenharia de Atributos está aplicando arredondamento para adequar o pré processamento.
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()
        for column in ORDINAL_COLUMNS:
            data[f"{column}_rounded"] = data[column].round()

        return data


# Função que faz a carga dos dados do CSV para o Dataset. Também remove duplicatas, se caso existir.
def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [column.strip() for column in df.columns]

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in the CSV.")

    return df.drop_duplicates().reset_index(drop=True)


# FUNÇÃO DO PRÉ PROCESSAMENTO..
# As colunas numéricas são imputadas usando a mediana, e opcionalmente escaladas. As colunas categóricas são imputadas usando a moda e codificadas usando one-hot encoding.
def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_columns = RAW_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_columns),
            ("categorical", categorical_pipeline, RAW_CATEGORICAL_COLUMNS),
        ]
    )


# FUNÇÃO DE UTILIZAÇÃO DOS MODELOS DE TREINAMENTO.
# Costrui dois pipelines. A Regração Logísticas e Random Florest.
def build_candidate_pipelines() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("features", FeatureEngineer()),
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                ("model", LogisticRegression(max_iter=3000, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("features", FeatureEngineer()),
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                ("model", RandomForestClassifier(n_estimators=400, random_state=42)),
            ]
        ),
    }

# FUNÇÃO DE AVALIAÇÃO DOS MODELOS.
# Utiliza validação cruzada estratificada e coleta informaç~~oes de média e desvio padrão para acurácia.
def evaluate_candidates(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[str, dict[str, dict[str, float]]]:
    candidates = build_candidate_pipelines()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: dict[str, dict[str, float]] = {}

    for name, pipeline in candidates.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=splitter, scoring="accuracy")
        results[name] = {
            "cv_mean_accuracy": float(scores.mean()),
            "cv_std_accuracy": float(scores.std()),
        }

    best_name = max(results, key=lambda model_name: results[model_name]["cv_mean_accuracy"])
    return best_name, results


# FUNÇÃO DE TESTE E TREINAMENTO DO MODELO.
# Utiliza função para avaliar o modelo utilizando validação cruzada e tamanho de teste 20% e as coletas das métricas de acurácia.
def train_and_evaluate(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    best_name, candidate_metrics = evaluate_candidates(X_train, y_train)
    best_pipeline = build_candidate_pipelines()[best_name]
    best_pipeline.fit(X_train, y_train)

    predictions = best_pipeline.predict(X_test)
    holdout_accuracy = accuracy_score(y_test, predictions)

    metrics = {
        "dataset_rows_after_deduplication": int(len(df)),
        "target_column": TARGET_COLUMN,
        "best_model": best_name,
        "candidate_metrics": candidate_metrics,
        "holdout_accuracy": float(holdout_accuracy),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=best_pipeline.classes_).tolist(),
        "class_labels": list(best_pipeline.classes_),
    }

    try:
        full_predictions = best_pipeline.predict(X)
        proba = None
        if hasattr(best_pipeline, "predict_proba"):
            proba = best_pipeline.predict_proba(X)

        preds_df = X.copy()
        preds_df["_y_true"] = y
        preds_df["_y_pred"] = full_predictions

        if proba is not None:
            for idx, class_label in enumerate(best_pipeline.classes_):
                preds_df[f"prob_{class_label}"] = proba[:, idx]

        metrics["prediction_rows"] = int(len(preds_df))
        metrics["_preds_sample"] = preds_df.head(3).to_dict(orient="records")
    except Exception:
        preds_df = None

    return best_pipeline, metrics, preds_df


# FUNÇÃO DE SALVAMENTO DO MODELO.
# Utiliza a função para salvar e gravar o resultado das análises e uma pasta chamada Artefacts, utilizando o joblib para o modelo e json para as métricas.
def save_artifacts(model: Pipeline, metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "obesity_pipeline.joblib"
    metrics_path = output_dir / "metrics.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


# FUNÇÃO DE SALVAMENTO DAS PREDIÇÕES.
# Salva as predições em um arquivo CSV, podendo ser utilizado para geração de dashboards em ambiente externo e análise de resultados. Salva dentro da pasta Artefacts.
def save_predictions(preds_df: pd.DataFrame | None, output_dir: Path) -> None:
    if preds_df is None:
        return

    preds_path = output_dir / "predictions.csv"
    preds_df.to_csv(preds_path, index=False, encoding="utf-8")


# FUNÇÃO DE ARGUMENTOS.
# Função para executar o script python não somente no notebook, mas também em ambiente CMD (Command Line). É possivel invocar parametros para execução.
# OBRIGATÓRIO PASSAR O CAMINHO DO ARQUIVO CSV.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de classificação do modelo para obesidade.")
    parser.add_argument("--data", required=True, help="Caminho do arquivo de Dataset(CSV).")
    parser.add_argument(
        "--output-dir",
        default="artefatos",
        help="Diretório onde será salvo os artefatos.",
    )
    return parser.parse_args()


# EXECUÇÃO DO PROCESSO MAIN.
def main() -> None:
    args = parse_args()
    csv_path = Path(args.data)
    output_dir = Path(args.output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")

    df = load_dataset(csv_path)
    model, metrics, preds_df = train_and_evaluate(df)
    save_artifacts(model, metrics, output_dir)
    save_predictions(preds_df, output_dir)

    print("----- Treinamento Concluído com Sucesso -----")
    print(f"Melhor Modelo avaliado: {metrics['best_model']}")
    print(f"Acurácia Atingida: {metrics['holdout_accuracy']:.4f}")
    print(f"Diretório dos Artefatos salvo: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
