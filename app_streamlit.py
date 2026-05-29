from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


# ===== MAPEAMENTO DE TRADUÇÕES =====
PORTUGUESE_LABELS = {
    # Feature names (column display)
    "Gender": "Gênero",
    "Age": "Idade",
    "family_history": "Histórico familiar de excesso de peso",
    "FAVC": "Consumo frequente de alimentos calóricos",
    "FCVC": "Frequência de consumo de vegetais",
    "NCP": "Número de refeições principais",
    "CAEC": "Consumo de lanches entre refeições",
    "SMOKE": "Hábito de fumar",
    "CH2O": "Consumo diário de água",
    "SCC": "Monitora ingestão calórica",
    "FAF": "Frequência de atividade física",
    "TUE": "Tempo usando dispositivos eletrônicos",
    "CALC": "Consumo de bebida alcoólica",
    "MTRANS": "Meio de transporte",
    
    # Obesity classes
    "Insufficient_Weight": "Abaixo do peso",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso I",
    "Overweight_Level_II": "Sobrepeso II",
    "Obesity_Type_I": "Obesidade I",
    "Obesity_Type_II": "Obesidade II",
    "Obesity_Type_III": "Obesidade III",
    
    # Categorical values
    "Female": "Feminino",
    "Male": "Masculino",
    "yes": "Sim",
    "no": "Não",
    "Sometimes": "Às vezes",
    "Frequently": "Frequentemente",
    "Always": "Sempre",
    "Public_Transportation": "Transporte público",
    "Walking": "A pé",
    "Automobile": "Carro",
    "Motorbike": "Moto",
    "Bike": "Bicicleta",
    
    # Risk groups
    "Baixo risco": "Baixo risco",
    "Risco moderado": "Risco moderado",
    "Alto risco": "Alto risco",
}

def get_pt(label: str, default: str | None = None) -> str:
    """Get Portuguese translation for a label."""
    return PORTUGUESE_LABELS.get(label, default or label)


DEFAULT_MODEL_PATH = Path("artifacts/obesity_pipeline.joblib")
DEFAULT_METRICS_PATH = Path("artifacts/metrics.json")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #fbfcfe 0%, #f5f8fc 100%);
            }
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2.2rem;
            }
            .hero-card {
                padding: 1.1rem 1.2rem;
                border-radius: 1rem;
                background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
                color: white;
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.12);
                margin-bottom: 1rem;
            }
            .hero-card h1, .hero-card p {
                color: white !important;
                margin-bottom: 0;
            }
            .section-chip {
                display: inline-block;
                padding: 0.25rem 0.7rem;
                border-radius: 999px;
                background: rgba(255,255,255,0.15);
                border: 1px solid rgba(255,255,255,0.22);
                font-size: 0.82rem;
                margin-bottom: 0.7rem;
            }
            .insight-card {
                padding: 1rem 1rem 0.9rem 1rem;
                border-radius: 0.9rem;
                background: white;
                border: 1px solid #e5edf7;
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
            }
            .insight-label {
                font-size: 0.8rem;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.2rem;
            }
            .insight-value {
                font-size: 1.7rem;
                font-weight: 700;
                color: #0f172a;
                line-height: 1.1;
            }
            .insight-help {
                color: #475569;
                font-size: 0.88rem;
                margin-top: 0.35rem;
            }
            .soft-panel {
                padding: 0.9rem 1rem;
                border-radius: 0.9rem;
                background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
                border: 1px solid #e6eef7;
                margin-bottom: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Must be defined here so joblib can deserialize the saved model."""
    ORDINAL_COLUMNS = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]
    
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()
        for column in self.ORDINAL_COLUMNS:
            data[f"{column}_rounded"] = data[column].round()

        return data


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


@st.cache_data
def load_metrics(metrics_path: str):
    metrics_file = Path(metrics_path)

    if not metrics_file.exists():
        return None

    return json.loads(metrics_file.read_text(encoding="utf-8"))


@st.cache_data
def load_predictions(predictions_path: str):
    predictions_file = Path(predictions_path)

    if not predictions_file.exists():
        return None

    df = pd.read_csv(predictions_file)

    # Accept both original and translated CSV schemas.
    translated_to_internal = {
        "Genero": "Gender",
        "Idade": "Age",
        "Altura": "Height",
        "Peso": "Weight",
        "Historico_familiar_excesso_peso": "family_history",
        "Consumo_frequente_alimentos_caloricos": "FAVC",
        "Frequencia_consumo_vegetais": "FCVC",
        "Numero_refeicoes_principais": "NCP",
        "Consumo_lanches_entre_refeicoes": "CAEC",
        "Habito_fumar": "SMOKE",
        "Consumo_diario_agua": "CH2O",
        "Monitora_ingestao_calorica": "SCC",
        "Frequencia_atividade_fisica": "FAF",
        "Tempo_dispositivos_eletronicos": "TUE",
        "Consumo_bebida_alcoolica": "CALC",
        "Meio_transporte": "MTRANS",
        "classe_real": "_y_true",
        "classe_prevista": "_y_pred",
        "prob_Abaixo_do_peso": "prob_Insufficient_Weight",
        "prob_Peso_normal": "prob_Normal_Weight",
        "prob_Obesidade_I": "prob_Obesity_Type_I",
        "prob_Obesidade_II": "prob_Obesity_Type_II",
        "prob_Obesidade_III": "prob_Obesity_Type_III",
        "prob_Sobrepeso_I": "prob_Overweight_Level_I",
        "prob_Sobrepeso_II": "prob_Overweight_Level_II",
    }
    df = df.rename(columns=translated_to_internal)

    reverse_value_map = {
        "Feminino": "Female",
        "Masculino": "Male",
        "Sim": "yes",
        "Nao": "no",
        "As_vezes": "Sometimes",
        "Frequentemente": "Frequently",
        "Sempre": "Always",
        "Transporte_publico": "Public_Transportation",
        "A_pe": "Walking",
        "Carro": "Automobile",
        "Moto": "Motorbike",
        "Bicicleta": "Bike",
        "Abaixo_do_peso": "Insufficient_Weight",
        "Peso_normal": "Normal_Weight",
        "Sobrepeso_I": "Overweight_Level_I",
        "Sobrepeso_II": "Overweight_Level_II",
        "Obesidade_I": "Obesity_Type_I",
        "Obesidade_II": "Obesity_Type_II",
        "Obesidade_III": "Obesity_Type_III",
    }

    for col in ["Gender", "family_history", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS", "_y_true", "_y_pred"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(reverse_value_map)

    return df


def render_training_results(metrics: dict) -> None:
    st.subheader("Resultados do treino")

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Melhor modelo", metrics["best_model"])
    summary_col2.metric("Acurácia de holdout", f"{metrics['holdout_accuracy']:.4f}")
    summary_col3.metric("Linhas pós-deduplicação", metrics["dataset_rows_after_deduplication"])

    candidate_rows = []
    for model_name, model_metrics in metrics["candidate_metrics"].items():
        candidate_rows.append(
            {
                "Modelo": model_name,
                "Acurácia CV (Média)": round(model_metrics["cv_mean_accuracy"], 4),
                "Acurácia CV (Desvio)": round(model_metrics["cv_std_accuracy"], 4),
                "Selecionado": model_name == metrics["best_model"],
            }
        )

    st.markdown("#### Comparação entre candidatos")
    st.dataframe(pd.DataFrame(candidate_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Relatório de classificação")
    classification_report = pd.DataFrame(metrics["classification_report"]).T
    # Translate class names
    classification_report.index = classification_report.index.map(lambda x: get_pt(x, x))
    st.dataframe(classification_report, use_container_width=True)

    st.markdown("#### Matriz de confusão")
    labels = metrics["class_labels"]
    labels_pt = [get_pt(label, label) for label in labels]
    confusion_frame = pd.DataFrame(
        metrics["confusion_matrix"],
        index=pd.Index(labels_pt, name="Real"),
        columns=pd.Index(labels_pt, name="Previsto"),
    )
    st.dataframe(confusion_frame, use_container_width=True)


def build_input_frame() -> pd.DataFrame:
    st.subheader("Formulário de dados do paciente")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(get_pt("Gender"), [get_pt("Female"), get_pt("Male")])
        age = st.number_input(get_pt("Age"), min_value=14.0, max_value=100.0, value=25.0, step=0.2)
        family_history = st.selectbox(get_pt("family_history"), [get_pt("no"), get_pt("yes")])
        favc = st.selectbox(get_pt("FAVC"), [get_pt("no"), get_pt("yes")])
        fcvc = st.slider(get_pt("FCVC"), min_value=1.0, max_value=3.0, value=2.0, step=0.2, help="1 - As vezes come vegetais\n| 2 - Frequentemente come vegetais\n| 3 - Sempre come vegetais")
        ncp = st.slider(get_pt("NCP"), min_value=1.0, max_value=4.0, value=4.0, step=1.0, help="1 - Uma Refeição | 2 - Duas Refeições | 3 - Três Refeições | 4 - Quatro ou mais refeições")

    with col2:
        caec = st.selectbox(get_pt("CAEC"), [get_pt("no"), get_pt("Sometimes"), get_pt("Frequently"), get_pt("Always")])
        smoke = st.selectbox(get_pt("SMOKE"), [get_pt("no"), get_pt("yes")])
        ch2o = st.slider(get_pt("CH2O"), min_value=1.0, max_value=3.0, value=2.0, step=0.2, help="1 - Até 1 Litro | 2 - 1-2 Litros p/dia | 3 - Mais de 2 litros p/dia")
        scc = st.selectbox(get_pt("SCC"), [get_pt("no"), get_pt("yes")])
        faf = st.slider(get_pt("FAF"), min_value=0.0, max_value=3.0, value=1.0, step=0.1)
        tue = st.slider(get_pt("TUE"), min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        calc = st.selectbox(get_pt("CALC"), [get_pt("no"), get_pt("Sometimes"), get_pt("Frequently"), get_pt("Always")])
        mtrans = st.selectbox(
            get_pt("MTRANS"),
            [get_pt("Public_Transportation"), get_pt("Walking"), get_pt("Automobile"), get_pt("Motorbike"), get_pt("Bike")],
        )
    
    # Reverse mapping for form input
    reverse_map = {v: k for k, v in PORTUGUESE_LABELS.items()}
    
    return pd.DataFrame(
        [
            {
                "Gender": reverse_map.get(gender, gender),
                "Age": age,
                "family_history": reverse_map.get(family_history, family_history),
                "FAVC": reverse_map.get(favc, favc),
                "FCVC": fcvc,
                "NCP": ncp,
                "CAEC": reverse_map.get(caec, caec),
                "SMOKE": reverse_map.get(smoke, smoke),
                "CH2O": ch2o,
                "SCC": reverse_map.get(scc, scc),
                "FAF": faf,
                "TUE": tue,
                "CALC": reverse_map.get(calc, calc),
                "MTRANS": reverse_map.get(mtrans, mtrans),
            }
        ]
    )


def map_risk_group(label: str) -> str:
    value = str(label).lower()

    if "insufficient" in value or "normal" in value or "underweight" in value:
        return "Baixo risco"

    if "overweight" in value:
        return "Risco moderado"

    if "obesity" in value or "obese" in value:
        return "Alto risco"

    return "Outro"


def prepare_analytics_frame(preds_df: pd.DataFrame) -> pd.DataFrame:
    df = preds_df.copy()
    df["risk_group"] = df["_y_pred"].map(map_risk_group)
    df["true_risk_group"] = df["_y_true"].map(map_risk_group)
    df["obesity_level"] = df["_y_pred"].astype(str)
    df["age_band"] = pd.cut(
        df["Age"],
        bins=[13, 19, 29, 39, 49, 59, 120],
        labels=["14-19", "20-29", "30-39", "40-49", "50-59", "60+"],
        include_lowest=True,
    )
    df["behavior_score"] = df[["FCVC", "CH2O", "FAF", "TUE"]].mean(axis=1)
    df["behavior_band"] = pd.cut(
        df["behavior_score"],
        bins=[0, 1.5, 2.25, 3.0],
        labels=["Baixo", "Médio", "Alto"],
        include_lowest=True,
    )
    df = df.rename(columns={"Grupo de Risco": "risk_group", "Faixa Etária": "age_band", "Escore Comportamental": "behavior_score", "Faixa Comportamental": "behavior_band"})
    return df


def render_dashboard_intro() -> None:
    st.markdown(
        """
        <div class='soft-panel'>
            <div class='section-chip'>Analytics Clínico</div>
            <h3 style='margin: 0 0 0.35rem 0; color: #0f172a;'>Leitura executiva para tomada de decisão</h3>
            <div style='color: #475569;'>
                Esta página resume distribuição de risco, fatores comportamentais, segmentação clínica e a evolução
                das predições por faixa etária para apoiar a equipe médica.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class='insight-card'>
            <div class='insight-label'>{label}</div>
            <div class='insight-value'>{value}</div>
            <div class='insight-help'>{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_analytics_insights(df: pd.DataFrame, metrics: dict | None) -> dict[str, str]:
    top_level = df["obesity_level"].value_counts().idxmax()
    top_level_share = df["obesity_level"].value_counts(normalize=True).max() * 100
    high_risk_share = (df["risk_group"] == "Alto risco").mean() * 100
    strongest_age_band = df.groupby("age_band")["risk_group"].apply(lambda s: (s == "Alto risco").mean()).idxmax()
    worst_true_pred_gap = (
        pd.crosstab(df["true_risk_group"], df["risk_group"], normalize="index")
        .reindex(index=["Baixo risco", "Risco moderado", "Alto risco"], fill_value=0)
    )

    return {
        "top_level": get_pt(top_level, top_level),
        "top_level_share": f"{top_level_share:.1f}%",
        "high_risk_share": f"{high_risk_share:.1f}%",
        "strongest_age_band": str(strongest_age_band),
        "best_model": metrics["best_model"] if metrics is not None else "N/D",
        "holdout_accuracy": f"{metrics['holdout_accuracy']:.4f}" if metrics is not None else "N/D",
        "gap_note": f"Maior concentração de alto risco em {strongest_age_band}.",
    }


def render_analytics_overview(df: pd.DataFrame, metrics: dict | None) -> None:
    st.subheader("Panorama geral")

    total_rows = len(df)
    unique_classes = df["_y_pred"].nunique()
    high_risk_share = (df["risk_group"] == "Alto risco").mean() * 100
    avg_age = df["Age"].mean()
    top_level = df["obesity_level"].value_counts().idxmax()
    top_level_share = df["obesity_level"].value_counts(normalize=True).max() * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_insight_card("Registros", f"{total_rows:,}", "Quantidade de predições analisadas")
    with k2:
        render_insight_card("Classes", str(unique_classes), "Número de níveis previstos")
    with k3:
        render_insight_card("Alto risco", f"{high_risk_share:.1f}%", "Participação de casos críticos")
    with k4:
        render_insight_card("Idade média", f"{avg_age:.1f}", "Média da população analisada")
    with k5:
        render_insight_card("Classe líder", get_pt(top_level, top_level), f"{top_level_share:.1f}% da base")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Distribuição das classes previstas")
        pred_counts = df["_y_pred"].value_counts().sort_values(ascending=False)
        pred_counts_pt = pred_counts.copy()
        pred_counts_pt.index = pred_counts_pt.index.map(lambda x: get_pt(x, x))
        st.bar_chart(pred_counts_pt)

    with right:
        st.markdown("#### Distribuição por faixa de risco")
        risk_counts = df["risk_group"].value_counts()
        st.bar_chart(risk_counts)

    st.markdown("#### Evolução das predições por nível de obesidade")
    evolution_by_level = (
        pd.crosstab(df["age_band"], df["obesity_level"], normalize="index")
        .sort_index()
        .fillna(0)
        .round(3)
    )
    # Translate column names for display
    evolution_by_level.columns = evolution_by_level.columns.map(lambda x: get_pt(x, x))
    st.line_chart(evolution_by_level)
    st.caption("Cada linha mostra a participação relativa de uma classe prevista em cada faixa etária.")

    st.markdown("#### Principais leituras")
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    insight_col1.info(f"Maior concentração em: {get_pt(top_level, top_level)}")
    insight_col2.info(f"Casos de alto risco: {high_risk_share:.1f}%")
    insight_col3.info(f"Idade média do painel: {avg_age:.0f} anos")

    if metrics is not None:
        st.caption(
            f"Melhor modelo treinado: {metrics['best_model']} | Acurácia de holdout: {metrics['holdout_accuracy']:.4f}"
        )


def render_profile_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Perfil de risco")

    st.markdown("<div class='soft-panel'>Comparações para priorizar grupos com maior exposição ao risco.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Faixa etária x risco")
        age_risk = pd.crosstab(df["age_band"], df["risk_group"])
        st.bar_chart(age_risk)

        st.markdown("#### Histórico familiar x risco")
        family_risk = pd.crosstab(df["family_history"], df["risk_group"], normalize="index").round(2)
        family_risk.index = family_risk.index.map(lambda x: get_pt(x, x))
        st.dataframe(family_risk, use_container_width=True)

    with col2:
        st.markdown("#### Sexo x risco")
        gender_risk = pd.crosstab(df["Gender"], df["risk_group"], normalize="index").round(2)
        gender_risk.index = gender_risk.index.map(lambda x: get_pt(x, x))
        st.dataframe(gender_risk, use_container_width=True)

        st.markdown("#### Consumo alimentar (FAVC) x risco")
        favc_risk = pd.crosstab(df["FAVC"], df["risk_group"], normalize="index").round(2)
        favc_risk.index = favc_risk.index.map(lambda x: get_pt(x, x))
        st.dataframe(favc_risk, use_container_width=True)

    st.markdown("#### Evolução do risco por faixa etária")
    age_risk_share = pd.crosstab(df["age_band"], df["risk_group"], normalize="index").fillna(0).round(3)
    st.line_chart(age_risk_share)
    st.caption("Mostra como a composição de risco evolui entre as faixas etárias.")


def render_behavior_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Fatores comportamentais")

    st.markdown("<div class='soft-panel'>Comparação dos hábitos que mais ajudam a explicar o perfil de obesidade.</div>", unsafe_allow_html=True)

    feature_cols = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    means_by_risk = df.groupby("risk_group")[feature_cols].mean().T
    # Translate feature names
    means_by_risk.index = means_by_risk.index.map(lambda x: get_pt(x, x))

    st.markdown("#### Média dos fatores por faixa de risco")
    st.bar_chart(means_by_risk)

    st.markdown("#### Tabela comparativa dos fatores")
    st.dataframe(means_by_risk.round(2), use_container_width=True)

    score_table = (
        df.groupby("risk_group")["behavior_score"]
        .agg(["mean", "median", "count"])
        .sort_values("mean", ascending=False)
        .round(2)
    )
    st.markdown("#### Escore comportamental agregado")
    st.dataframe(score_table, use_container_width=True)

def render_segmentation_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Segmentação clínica")

    st.markdown("<div class='soft-panel'>Visão rápida de como os casos se distribuem entre faixas de severidade.</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    low_share = (df["risk_group"] == "Baixo risco").mean() * 100
    moderate_share = (df["risk_group"] == "Risco moderado").mean() * 100
    high_share = (df["risk_group"] == "Alto risco").mean() * 100

    col1.metric("Baixo risco", f"{low_share:.1f}%")
    col2.metric("Risco moderado", f"{moderate_share:.1f}%")
    col3.metric("Alto risco", f"{high_share:.1f}%")

    segmentation_table = pd.crosstab(df["true_risk_group"], df["risk_group"], normalize="index").round(2)
    st.markdown("#### Matriz de segmentação real x prevista")
    st.dataframe(segmentation_table, use_container_width=True)

    st.markdown("#### Segmentação por faixa etária")
    age_segment = pd.crosstab(df["age_band"], df["risk_group"], normalize="index").round(2)
    st.dataframe(age_segment, use_container_width=True)

    st.markdown("#### Linha de segmentação por faixa etária")
    st.line_chart(age_segment)


def render_correlation_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Correlação e relações entre variáveis")

    st.markdown("<div class='soft-panel'>Ajuda a identificar quais variáveis se movem juntas e onde há redundância analítica.</div>", unsafe_allow_html=True)

    numeric_columns = [column for column in ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE", "behavior_score"] if column in df.columns]
    corr = df[numeric_columns].corr().round(2)
    # Translate column names for display
    corr.index = corr.index.map(lambda x: get_pt(x, x))
    corr.columns = corr.columns.map(lambda x: get_pt(x, x))
    
    st.markdown("#### Matriz de correlação")
    st.dataframe(corr, use_container_width=True)
    st.caption("Se quiser o efeito visual em gradiente, instale matplotlib ou aplique a formatação fora do Streamlit.")

    st.markdown("#### Relação entre idade e comportamento")
    scatter_frame = df[["Age", "behavior_score", "risk_group"]].copy()
    st.scatter_chart(scatter_frame, x="Age", y="behavior_score", color=None)

    st.markdown("#### Evolução da correlação com a idade")
    age_profile = (
        df.groupby("age_band")[["FCVC", "CH2O", "FAF", "TUE", "behavior_score"]]
        .mean()
        .round(2)
    )
    # Translate column names for display
    age_profile.columns = age_profile.columns.map(lambda x: get_pt(x, x))
    st.line_chart(age_profile)


def render_analytics_page(preds_df: pd.DataFrame | None, metrics: dict | None) -> None:
    st.header("Visões analíticas")
    render_dashboard_intro()

    if preds_df is None or preds_df.empty:
        st.info("Arquivo de predições não encontrado em artifacts/predictions.csv. Rode o treino novamente.")
        return

    df = prepare_analytics_frame(preds_df)
    insights = build_analytics_insights(df, metrics)

    st.markdown("#### Insights-chave")
    insight_a, insight_b, insight_c = st.columns(3)
    with insight_a:
        render_insight_card("Classe mais frequente", insights["top_level"], insights["top_level_share"])
    with insight_b:
        render_insight_card("Casos de alto risco", insights["high_risk_share"], "Proporção total de perfis críticos")
    with insight_c:
        render_insight_card("Faixa etária crítica", insights["strongest_age_band"], "Maior concentração de alto risco")

    tabs = st.tabs([
        "Panorama geral",
        "Perfil de risco",
        "Fatores comportamentais",
        "Segmentação clínica",
        "Análise de correlação",
    ])

    with tabs[0]:
        render_analytics_overview(df, metrics)

    with tabs[1]:
        render_profile_dashboard(df)

    with tabs[2]:
        render_behavior_dashboard(df)

    with tabs[3]:
        render_segmentation_dashboard(df)

    with tabs[4]:
        render_correlation_dashboard(df)

    st.markdown("---")
    st.subheader("Recomendações clínicas")
    st.markdown(
        f"""
        - Priorizar acompanhamento dos pacientes concentrados em **alto risco**, especialmente na faixa **{insights['strongest_age_band']}**.
        - Direcionar ações educativas para os grupos com menor escore comportamental e maior presença de classes severas.
        - Monitorar a classe dominante do painel, atualmente **{insights['top_level']}**, para definir estratégias de triagem.
        - Se o modelo treinado for usado operacionalmente, acompanhar a acurácia de holdout (**{insights['holdout_accuracy']}**) como referência de estabilidade.
        """
    )


def main() -> None:
    st.set_page_config(page_title="Predição de Obesidade", layout="wide")
    inject_styles()
    st.markdown(
        """
        <div class='hero-card'>
            <div class='section-chip'>Inteligência Médica de Obesidade</div>
            <h1><b>OBESIDADE: Formulário e Análises Clínicas</b></h1>
            <p>Este site apresenta um formulário para previsão individual dos pacientes para prever o nível de obesidade com base nas informações</p>
            <p>Além disso, inclui uma seção de análises para explorar as predições em conjunto, identificar padrões de risco e apoiar decisões clínicas a respeito.</p>
            <p>O Conteúdo contém informações do Modelo Utilizado, Métricas de treinamento e exemplo dos dados brutos utilizados.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_path = st.sidebar.text_input("Caminho do modelo", str(DEFAULT_MODEL_PATH))
    metrics_path = st.sidebar.text_input("Caminho das métricas", str(DEFAULT_METRICS_PATH))
    model_file = Path(model_path)
    metrics_file = Path(metrics_path)

    if not model_file.exists():
        st.warning(
            "Modelo não encontrado. Por favor, execute o 'train_model.py' antes de executar o Streamlit. "
            "Caso foi treinado e não carregou, verifique o caminho informado dos artefatos na barra lateral."
        )
        st.stop()

    model = load_model(str(model_file))
    metrics = load_metrics(str(metrics_file))

    preds_df = load_predictions("artifacts/predictions.csv")

    page = st.sidebar.selectbox("Navegação", ["Predição", "Análise", "Sobre o modelo"])

    if page == "Predição":
        st.header("Formulário de Dados do Paciente")
        input_frame = build_input_frame()

        if st.button("Realizar predição", type="primary", icon="🔍"):
            prediction = model.predict(input_frame)[0]
            probabilities = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_frame)[0]

            st.success(f"Classe prevista: {get_pt(prediction, prediction)}")

            if probabilities is not None:
                probability_frame = pd.DataFrame(
                    {
                        "Nível de Obesidade": [get_pt(c, c) for c in model.classes_],
                        "Probabilidade": probabilities,
                    }
                ).sort_values("Probabilidade", ascending=False)

                probability_frame["Probabilidade"] = probability_frame["Probabilidade"].map(lambda value: round(value, 4))
                st.subheader("Probabilidades por classe")
                st.dataframe(probability_frame, use_container_width=True, hide_index=True)

    elif page == "Análise":
        render_analytics_page(preds_df, metrics)

    else:  # Sobre o modelo
        st.header("Informações do modelo")
        st.subheader("Resumo da arquitetura")
        st.text(str(model))

        if metrics is not None:
            st.subheader("Métricas de treino e validação")
            st.json(metrics)
        else:
            st.info("Nenhuma métrica encontrada. Rode o treino para gerar métricas.")


if __name__ == "__main__":
    main()
