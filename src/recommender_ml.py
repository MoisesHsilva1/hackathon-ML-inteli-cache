"""
recommender_ml.py
SageMaker Processing Job — Cache Warmup Recommender (ML-Enhanced)

Modelo híbrido:
  1. Matrix Factorization (ALS / SVD) sobre a matriz customer×document
  2. Score de negócio (volume, recência, peso qualitativo)
  3. Score final = w_mf · score_mf + w_biz · score_biz

Entrada : s3://<BUCKET>/input/*.csv  (logs Polaris, separador |)
Saída   : s3://<BUCKET>/output/warmup/warmup_recommendations.json

Autor   : Cache Warmup Recommender v2.0 (ML-Enhanced)
"""

import json, os, logging, time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Parâmetros via env vars ──────────────────────────────────────────────────
# Pesos do score híbrido
ALPHA        = float(os.environ.get("ALPHA",        0.35))   # peso volume
BETA         = float(os.environ.get("BETA",         0.40))   # peso recência
GAMMA        = float(os.environ.get("GAMMA",        0.25))   # peso negócio
LAMBDA_DECAY = float(os.environ.get("LAMBDA_DECAY", 0.15))   # taxa decaimento

# Pesos do modelo híbrido (MF vs Business)
W_MF         = float(os.environ.get("W_MF",         0.45))   # peso matrix factorization
W_BIZ        = float(os.environ.get("W_BIZ",        0.55))   # peso score de negócio

# Parâmetros do SVD
N_COMPONENTS = int(os.environ.get("N_COMPONENTS",   50))      # dimensões latentes
N_ITER       = int(os.environ.get("N_ITER",         15))      # iterações SVD

# Top-N por dimensão
TOP_N_REPORTS   = int(os.environ.get("TOP_N_REPORTS",   20))
TOP_N_FEATURES  = int(os.environ.get("TOP_N_FEATURES",  30))
TOP_N_CUSTOMERS = int(os.environ.get("TOP_N_CUSTOMERS", 500))
TOP_N_CDOCS     = int(os.environ.get("TOP_N_CDOCS",    1000))
TOP_N_PAIRS     = int(os.environ.get("TOP_N_PAIRS",    2000))

# Caminhos SageMaker
INPUT_DIR  = os.environ.get("SM_INPUT_DIR",  "/opt/ml/processing/input")
OUTPUT_DIR = os.environ.get("SM_OUTPUT_DIR", "/opt/ml/processing/output")
CSV_SEP    = os.environ.get("CSV_SEP", "|")


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO 1: Preparação de Dados
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(input_dir: str) -> pd.DataFrame:
    """Carrega todos os CSVs do diretório de entrada."""
    csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {input_dir}")

    frames = []
    for f in csv_files:
        path = os.path.join(input_dir, f)
        log.info("Carregando %s", path)
        frames.append(pd.read_csv(path, sep=CSV_SEP, low_memory=False))

    df = pd.concat(frames, ignore_index=True)
    log.info("Total de registros carregados: %d", len(df))
    return df


def prepare(df: pd.DataFrame) -> tuple:
    """Limpa e prepara o DataFrame."""
    df = df.copy()

    # Converte tipos
    df["inclusionDate"]    = pd.to_datetime(df["inclusionDate"], utc=False)
    df["billing"]          = df["billing"].astype(bool).astype(int)
    df["inquiry"]          = df["inquiry"].astype(bool).astype(int)
    df["post_report_view"] = df["post_report_view"].astype(bool).astype(int)
    df["httpTime"]         = pd.to_numeric(df["httpTime"], errors="coerce").fillna(0)
    df["httpStatus"]       = pd.to_numeric(df["httpStatus"], errors="coerce").fillna(0)

    # Limpa strings (aspas)
    str_cols = ["reportName", "TYPE_REPORT", "FEATURENAME", "FEATURE_TYPE",
                "customerDocument", "consultedDocument"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.strip('"')

    # Data de referência e delta
    t_ref = df["inclusionDate"].max()
    df["delta_days"] = (t_ref - df["inclusionDate"]).dt.total_seconds() / 86400.0

    log.info("Período: %s → %s (%d dias)",
             df["inclusionDate"].min().date(), t_ref.date(),
             (t_ref - df["inclusionDate"].min()).days)

    return df, t_ref


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO 2: Matrix Factorization (Collaborative Filtering via SVD)
# ═══════════════════════════════════════════════════════════════════════════════

def build_interaction_matrix(df: pd.DataFrame):
    """
    Constrói a matriz de interações customer × consultedDocument.
    Valores = frequência de acesso (implicit feedback).
    """
    log.info("Construindo matriz de interações...")

    # Codifica IDs em inteiros
    customers = df["customerDocument"].astype(str).unique()
    documents = df["consultedDocument"].astype(str).unique()

    cust_to_idx = {c: i for i, c in enumerate(customers)}
    doc_to_idx  = {d: i for i, d in enumerate(documents)}

    # Conta interações por par
    interactions = df.groupby(["customerDocument", "consultedDocument"]).size().reset_index(name="count")

    rows = interactions["customerDocument"].astype(str).map(cust_to_idx).values
    cols = interactions["consultedDocument"].astype(str).map(doc_to_idx).values
    vals = interactions["count"].values.astype(np.float32)

    # Aplica log1p para suavizar distribuição (implicit feedback)
    vals = np.log1p(vals)

    matrix = csr_matrix((vals, (rows, cols)), shape=(len(customers), len(documents)))

    log.info("Matriz: %d customers × %d documents | %d interações não-zero",
             matrix.shape[0], matrix.shape[1], matrix.nnz)

    return matrix, cust_to_idx, doc_to_idx, customers, documents


def train_svd(matrix: csr_matrix, n_components: int, n_iter: int):
    """Treina o modelo SVD (Truncated SVD para matrizes esparsas)."""
    log.info("Treinando SVD com %d componentes, %d iterações...", n_components, n_iter)

    # Ajusta n_components se a matriz for menor
    actual_components = min(n_components, matrix.shape[0] - 1, matrix.shape[1] - 1)

    t0 = time.time()
    svd = TruncatedSVD(n_components=actual_components, n_iter=n_iter, random_state=42)
    user_factors = svd.fit_transform(matrix)       # (n_users, n_components)
    item_factors = svd.components_.T               # (n_items, n_components)

    elapsed = time.time() - t0
    explained_var = svd.explained_variance_ratio_.sum()

    log.info("SVD treinado em %.1fs | Variância explicada: %.2f%% | Componentes: %d",
             elapsed, explained_var * 100, actual_components)

    return svd, user_factors, item_factors, explained_var


def evaluate_model(matrix: csr_matrix, svd, user_factors, item_factors):
    """Avalia o modelo com métricas de reconstrução."""
    # Reconstrói uma amostra da matriz
    n_sample = min(5000, matrix.shape[0])
    sample_idx = np.random.choice(matrix.shape[0], n_sample, replace=False)

    reconstructed = user_factors[sample_idx] @ item_factors.T
    original = matrix[sample_idx].toarray()

    # RMSE da reconstrução (em escala log1p)
    mask = original > 0
    if mask.sum() > 0:
        rmse = np.sqrt(np.mean((reconstructed[mask] - original[mask]) ** 2))
    else:
        rmse = 0.0

    log.info("Avaliação (amostra %d users): RMSE reconstrução = %.4f", n_sample, rmse)
    return rmse


def get_mf_scores_for_pairs(df: pd.DataFrame, user_factors, item_factors,
                            cust_to_idx, doc_to_idx) -> pd.DataFrame:
    """Calcula score MF para os pares observados no dataset."""
    pairs = df.groupby(["customerDocument", "consultedDocument"]).size().reset_index(name="count")

    scores = []
    for _, row in pairs.iterrows():
        c = str(row["customerDocument"])
        d = str(row["consultedDocument"])
        if c in cust_to_idx and d in doc_to_idx:
            u_vec = user_factors[cust_to_idx[c]]
            i_vec = item_factors[doc_to_idx[d]]
            score = float(np.dot(u_vec, i_vec))
        else:
            score = 0.0
        scores.append(score)

    pairs["mf_score_raw"] = scores

    # Normaliza para [0, 1]
    scaler = MinMaxScaler()
    pairs["mf_score"] = scaler.fit_transform(pairs[["mf_score_raw"]])

    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO 3: Score de Negócio (Heurístico)
# ═══════════════════════════════════════════════════════════════════════════════

AGG_SPEC = dict(
    volume       =("billing",          "count"),
    delta_mean   =("delta_days",       "mean"),
    billing_mean =("billing",          "mean"),
    inquiry_mean =("inquiry",          "mean"),
    view_mean    =("post_report_view", "mean"),
    httpTime_mean=("httpTime",         "mean"),
)


def score_business(df: pd.DataFrame, group_cols: list, max_vol: int, max_lat: float) -> pd.DataFrame:
    """Calcula o score de negócio por agrupamento."""
    agg = df.groupby(group_cols).agg(**AGG_SPEC).reset_index()

    agg["V_norm"] = agg["volume"] / max_vol
    agg["R"]      = np.exp(-LAMBDA_DECAY * agg["delta_mean"])
    agg["W"]      = (0.30 * agg["billing_mean"]
                   + 0.25 * agg["inquiry_mean"]
                   + 0.25 * agg["view_mean"]
                   + 0.20 * (agg["httpTime_mean"] / max_lat))
    agg["biz_score"] = ALPHA * agg["V_norm"] + BETA * agg["R"] + GAMMA * agg["W"]

    return agg.sort_values("biz_score", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO 4: Score Híbrido
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hybrid_pairs(df, user_factors, item_factors, cust_to_idx, doc_to_idx,
                         max_vol, max_lat):
    """Combina MF score + Business score para pares customer→document."""
    log.info("Calculando score híbrido para pares...")

    # Score de negócio para pares
    biz = score_business(df, ["customerDocument", "consultedDocument"], max_vol, max_lat)

    # Score MF para pares
    mf_pairs = get_mf_scores_for_pairs(df, user_factors, item_factors, cust_to_idx, doc_to_idx)

    # Merge
    merged = biz.merge(
        mf_pairs[["customerDocument", "consultedDocument", "mf_score"]],
        on=["customerDocument", "consultedDocument"],
        how="left"
    )
    merged["mf_score"] = merged["mf_score"].fillna(0)

    # Score híbrido
    merged["score"] = W_MF * merged["mf_score"] + W_BIZ * merged["biz_score"]

    return merged.sort_values("score", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO 5: Geração de Recomendações
# ═══════════════════════════════════════════════════════════════════════════════

def to_recs(df_dim: pd.DataFrame, cols: list, n: int, score_col: str = "biz_score") -> list:
    """Converte DataFrame em lista de recomendações."""
    rows = []
    for r in df_dim[cols + ["volume", score_col]].head(n).itertuples(index=False):
        rec = {}
        for c in cols:
            rec[c] = str(getattr(r, c))
        rec["volume"] = int(getattr(r, "volume"))
        rec["score"]  = round(float(getattr(r, score_col)), 4)
        rows.append(rec)
    return rows


def generate_recommendations(df, hybrid_pairs, t_ref, model_metrics):
    """Gera o payload final de recomendações."""
    max_vol = len(df)
    max_lat = df["httpTime"].max()

    # Dimensões com score de negócio
    rep  = score_business(df, ["reportName", "TYPE_REPORT"], max_vol, max_lat)
    feat = score_business(df, ["ID_FEATURE", "FEATURENAME", "FEATURE_TYPE"], max_vol, max_lat)
    cust = score_business(df, ["customerDocument"], max_vol, max_lat)
    cdoc = score_business(df, ["consultedDocument"], max_vol, max_lat)

    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "model_version": "2.0-hybrid-svd",
        "model_params": {
            "formula": "score_final = w_mf · score_mf + w_biz · (α·V_norm + β·exp(-λ·Δt) + γ·W_biz)",
            "W_biz_formula": "W = 0.30·billing + 0.25·inquiry + 0.25·post_view + 0.20·latency_norm",
            "alpha": ALPHA, "beta": BETA, "gamma": GAMMA, "lambda": LAMBDA_DECAY,
            "w_mf": W_MF, "w_biz": W_BIZ,
            "svd_components": N_COMPONENTS, "svd_iterations": N_ITER,
        },
        "model_metrics": model_metrics,
        "warmup_targets": {
            "top_reports":             to_recs(rep,  ["reportName", "TYPE_REPORT"], TOP_N_REPORTS),
            "top_features":            to_recs(feat, ["ID_FEATURE", "FEATURENAME", "FEATURE_TYPE"], TOP_N_FEATURES),
            "top_customers":           to_recs(cust, ["customerDocument"], TOP_N_CUSTOMERS),
            "top_consulted_documents": to_recs(cdoc, ["consultedDocument"], TOP_N_CDOCS),
            "top_pairs":               to_recs(hybrid_pairs, ["customerDocument", "consultedDocument"],
                                               TOP_N_PAIRS, score_col="score"),
        },
        "stats": {
            "total_records":    len(df),
            "unique_customers": int(df["customerDocument"].nunique()),
            "unique_consulted": int(df["consultedDocument"].nunique()),
            "unique_reports":   int(df["reportName"].nunique()),
            "unique_features":  int(df["FEATURENAME"].nunique()),
            "date_range": {
                "from": str(df["inclusionDate"].min().date()),
                "to":   str(t_ref.date()),
            },
        },
    }
    return payload


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO 6: Pipeline Principal
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    t_start = time.time()
    log.info("=" * 60)
    log.info("Cache Warmup Recommender v2.0 (ML-Enhanced)")
    log.info("=" * 60)

    # 1. Carregar dados
    df = load_data(INPUT_DIR)
    df, t_ref = prepare(df)
    max_vol = len(df)
    max_lat = df["httpTime"].max()

    # 2. Construir matriz de interações
    matrix, cust_to_idx, doc_to_idx, customers, documents = build_interaction_matrix(df)

    # 3. Treinar modelo SVD
    svd, user_factors, item_factors, explained_var = train_svd(matrix, N_COMPONENTS, N_ITER)

    # 4. Avaliar modelo
    rmse = evaluate_model(matrix, svd, user_factors, item_factors)

    model_metrics = {
        "svd_explained_variance_pct": round(explained_var * 100, 2),
        "svd_reconstruction_rmse": round(rmse, 4),
        "matrix_shape": list(matrix.shape),
        "matrix_density_pct": round(matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100, 4),
        "n_interactions": int(matrix.nnz),
        "training_time_seconds": None,  # preenchido abaixo
    }

    # 5. Score híbrido para pares
    hybrid_pairs = compute_hybrid_pairs(df, user_factors, item_factors,
                                        cust_to_idx, doc_to_idx, max_vol, max_lat)

    # 6. Gerar recomendações
    payload = generate_recommendations(df, hybrid_pairs, t_ref, model_metrics)

    elapsed = time.time() - t_start
    payload["model_metrics"]["training_time_seconds"] = round(elapsed, 1)
    log.info("Pipeline completo em %.1fs", elapsed)

    # 7. Salvar output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "warmup_recommendations.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    log.info("✅ Recomendações salvas em %s", out_path)

    # Log top-5 pares
    log.info("Top-5 pares recomendados:")
    for p in payload["warmup_targets"]["top_pairs"][:5]:
        log.info("  customer=%s → doc=%s | score=%.4f vol=%d",
                 p["customerDocument"], p["consultedDocument"], p["score"], p["volume"])

    return out_path


if __name__ == "__main__":
    run()
