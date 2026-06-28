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

import glob
import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

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
    """Load and concatenate all CSV files from the input directory.

    Args:
        input_dir: Path to directory containing pipe-separated CSV files.

    Returns:
        Concatenated DataFrame with all records.

    Raises:
        FileNotFoundError: When no CSV files are found in input_dir.
    """
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in directory: {input_dir}. "
            "Expected at least one .csv file for processing."
        )

    frames = []
    for path in csv_files:
        log.info("Loading %s", path)
        frames.append(pd.read_csv(path, sep=CSV_SEP, low_memory=False))

    df = pd.concat(frames, ignore_index=True)
    log.info("Total records loaded: %d from %d file(s)", len(df), len(csv_files))
    return df


def _to_bool_int(series: pd.Series) -> pd.Series:
    """Convert a series to boolean integer (0 or 1).

    Handles various input types: int 0/1, bool True/False,
    string "0"/"1"/"True"/"False", and numeric values.
    """
    def _coerce(val) -> int:
        if isinstance(val, bool):
            return int(val)
        if isinstance(val, (int, float)):
            if pd.isna(val):
                return 0
            return 1 if int(val) != 0 else 0
        s = str(val).strip().lower()
        if s in ("1", "true"):
            return 1
        return 0

    return series.map(_coerce).astype(int)


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Clean and prepare the DataFrame for model training.

    Performs type conversion, string cleaning, and computes delta_days.

    Args:
        df: Raw DataFrame loaded from CSV files.

    Returns:
        Tuple of (cleaned DataFrame, reference timestamp t_ref).
    """
    df = df.copy()

    # Convert types
    df["inclusionDate"] = pd.to_datetime(df["inclusionDate"], utc=False)
    df["billing"] = _to_bool_int(df["billing"])
    df["inquiry"] = _to_bool_int(df["inquiry"])
    df["post_report_view"] = _to_bool_int(df["post_report_view"])
    df["httpTime"] = pd.to_numeric(df["httpTime"], errors="coerce").fillna(0)
    df["httpStatus"] = pd.to_numeric(df["httpStatus"], errors="coerce").fillna(0)

    # Clean string columns (strip whitespace and quotation marks)
    str_cols = [
        "reportName", "TYPE_REPORT", "FEATURENAME", "FEATURE_TYPE",
        "customerDocument", "consultedDocument",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.strip('"')
                .str.strip("'")
                .str.strip()
            )

    # Reference date and delta_days
    t_ref = df["inclusionDate"].max()
    df["delta_days"] = (t_ref - df["inclusionDate"]).dt.total_seconds() / 86400.0

    log.info(
        "Period: %s → %s (%d days)",
        df["inclusionDate"].min().date(),
        t_ref.date(),
        (t_ref - df["inclusionDate"].min()).days,
    )

    return df, t_ref


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO 2: Matrix Factorization (Collaborative Filtering via SVD)
# ═══════════════════════════════════════════════════════════════════════════════

def build_interaction_matrix(
    df: pd.DataFrame,
) -> tuple[csr_matrix, dict[str, int], dict[str, int], np.ndarray, np.ndarray]:
    """Build sparse interaction matrix from access logs.

    Pipeline:
        1. Count interactions per (customerDocument, consultedDocument) pair
        2. Apply log1p to raw counts (dampens power-law distribution)
        3. Construct CSR sparse matrix
        4. Build bidirectional index mappings

    Args:
        df: Prepared DataFrame with customerDocument and consultedDocument columns.

    Returns:
        Tuple of (csr_matrix, cust_to_idx, doc_to_idx, customers_array, documents_array)
    """
    log.info("Construindo matriz de interações...")

    customers: np.ndarray = np.array(
        df["customerDocument"].astype(str).unique()
    )
    documents: np.ndarray = np.array(
        df["consultedDocument"].astype(str).unique()
    )

    cust_to_idx: dict[str, int] = {c: i for i, c in enumerate(customers)}
    doc_to_idx: dict[str, int] = {d: i for i, d in enumerate(documents)}

    # Conta interações por par (customerDocument, consultedDocument)
    interactions = (
        df.groupby(["customerDocument", "consultedDocument"])
        .size()
        .reset_index(name="count")
    )

    rows = interactions["customerDocument"].astype(str).map(cust_to_idx).values
    cols = interactions["consultedDocument"].astype(str).map(doc_to_idx).values
    vals = interactions["count"].values.astype(np.float32)

    # Aplica log1p para suavizar distribuição (implicit feedback)
    vals = np.log1p(vals)

    matrix = csr_matrix(
        (vals, (rows, cols)), shape=(len(customers), len(documents))
    )

    total_cells = matrix.shape[0] * matrix.shape[1]
    density_pct = (matrix.nnz / total_cells * 100) if total_cells > 0 else 0.0

    log.info(
        "Matriz: %d customers × %d documents | %d interações não-zero | "
        "densidade: %.4f%%",
        matrix.shape[0],
        matrix.shape[1],
        matrix.nnz,
        density_pct,
    )

    return matrix, cust_to_idx, doc_to_idx, customers, documents


def train_svd(
    matrix: csr_matrix, n_components: int, n_iter: int
) -> tuple[TruncatedSVD, np.ndarray, np.ndarray, float]:
    """Train TruncatedSVD on the interaction matrix.

    Automatically reduces n_components if it exceeds min(m-1, n-1).
    Uses random_state=42 for reproducibility.

    Args:
        matrix: CSR sparse interaction matrix of shape (m, n).
        n_components: Target latent dimensions (default 50).
        n_iter: Power iteration count (default 15).

    Returns:
        Tuple of (svd_model, user_factors, item_factors, explained_variance_ratio_sum).
    """
    m, n = matrix.shape
    actual_components = min(n_components, m - 1, n - 1)

    log.info(
        "Treinando SVD com %d componentes (solicitado: %d), %d iterações...",
        actual_components, n_components, n_iter,
    )

    t0 = time.time()
    svd = TruncatedSVD(
        n_components=actual_components, n_iter=n_iter, random_state=42
    )
    user_factors = svd.fit_transform(matrix)  # (m, actual_components)
    item_factors = svd.components_.T          # (n, actual_components)

    elapsed = time.time() - t0
    explained_var = float(svd.explained_variance_ratio_.sum())

    log.info(
        "SVD treinado em %.1fs | Variância explicada: %.2f%% | Componentes: %d",
        elapsed, explained_var * 100, actual_components,
    )

    return svd, user_factors, item_factors, explained_var


def evaluate_model(
    matrix: csr_matrix,
    svd: TruncatedSVD,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
) -> float:
    """Compute reconstruction RMSE on sampled non-zero entries.

    Samples up to 5000 users, reconstructs their vectors via
    user_factors @ item_factors.T, and measures RMSE only on positions
    where the original matrix has non-zero values (avoids sparse zero bias).

    Returns 0.0 if no non-zero entries exist in the evaluation sample.
    """
    n_sample = min(matrix.shape[0], 5000)

    if n_sample == matrix.shape[0]:
        sample_idx = np.arange(matrix.shape[0])
    else:
        sample_idx = np.random.choice(matrix.shape[0], n_sample, replace=False)

    reconstructed = user_factors[sample_idx] @ item_factors.T
    original = matrix[sample_idx].toarray()

    mask = original > 0
    if mask.sum() == 0:
        log.info("Avaliação (amostra %d users): sem entradas não-zero", n_sample)
        return 0.0

    rmse = float(np.sqrt(np.mean((reconstructed[mask] - original[mask]) ** 2)))

    log.info("Avaliação (amostra %d users): RMSE reconstrução = %.4f", n_sample, rmse)
    return rmse


def get_mf_scores_for_pairs(
    df: pd.DataFrame,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    cust_to_idx: dict[str, int],
    doc_to_idx: dict[str, int],
) -> pd.DataFrame:
    """Compute normalized MF scores for all observed customer-document pairs.

    For each unique (customerDocument, consultedDocument) pair in the dataset,
    computes the dot product of user_factor · item_factor. Assigns 0.0 when
    customer or document is absent from the index mappings. Normalizes all
    scores to [0, 1] using MinMaxScaler.

    Args:
        df: Prepared DataFrame with customerDocument and consultedDocument columns.
        user_factors: Array of shape (n_customers, n_components) from SVD.
        item_factors: Array of shape (n_documents, n_components) from SVD.
        cust_to_idx: Mapping from customer identifier to matrix row index.
        doc_to_idx: Mapping from document identifier to matrix column index.

    Returns:
        DataFrame with columns: customerDocument, consultedDocument, count,
        mf_score_raw, mf_score (normalized to [0, 1]).
    """
    pairs = (
        df.groupby(["customerDocument", "consultedDocument"])
        .size()
        .reset_index(name="count")
    )

    customers = pairs["customerDocument"].astype(str).values
    documents = pairs["consultedDocument"].astype(str).values

    scores = np.zeros(len(pairs), dtype=np.float64)

    for i in range(len(pairs)):
        c = customers[i]
        d = documents[i]
        if c in cust_to_idx and d in doc_to_idx:
            scores[i] = float(
                np.dot(user_factors[cust_to_idx[c]], item_factors[doc_to_idx[d]])
            )

    pairs["mf_score_raw"] = scores

    scaler = MinMaxScaler()
    pairs["mf_score"] = scaler.fit_transform(pairs[["mf_score_raw"]])

    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO 3: Score de Negócio (Heurístico)
# ═══════════════════════════════════════════════════════════════════════════════

AGG_SPEC: dict[str, tuple[str, str]] = {
    "volume":        ("delta_days",       "count"),
    "delta_mean":    ("delta_days",       "mean"),
    "billing_mean":  ("billing",          "mean"),
    "inquiry_mean":  ("inquiry",          "mean"),
    "view_mean":     ("post_report_view", "mean"),
    "httpTime_mean": ("httpTime",         "mean"),
}


def score_business(
    df: pd.DataFrame,
    group_cols: list[str],
    max_vol: int,
    max_lat: float,
) -> pd.DataFrame:
    """Compute business heuristic score for each group.

    Groups the DataFrame by ``group_cols`` and aggregates:
      - count → volume
      - mean of delta_days, billing, inquiry, post_report_view, httpTime

    Formula:
        biz_score = ALPHA * (volume / max_vol)
                  + BETA  * exp(-LAMBDA_DECAY * delta_mean)
                  + GAMMA * W_biz

    Where:
        W_biz = 0.30 * billing_mean
              + 0.25 * inquiry_mean
              + 0.25 * view_mean
              + 0.20 * (httpTime_mean / max_lat)

    Args:
        df: Prepared DataFrame with required columns.
        group_cols: Columns to group by before aggregation.
        max_vol: Maximum volume for normalization (typically len(df)).
        max_lat: Maximum latency for normalization (typically df["httpTime"].max()).

    Returns:
        Aggregated DataFrame with ``biz_score`` column, sorted descending.
    """
    agg = df.groupby(group_cols).agg(**AGG_SPEC).reset_index()

    agg["V_norm"] = agg["volume"] / max_vol
    agg["R"] = np.exp(-LAMBDA_DECAY * agg["delta_mean"])
    agg["W"] = (
        0.30 * agg["billing_mean"]
        + 0.25 * agg["inquiry_mean"]
        + 0.25 * agg["view_mean"]
        + 0.20 * (agg["httpTime_mean"] / max_lat)
    )
    agg["biz_score"] = ALPHA * agg["V_norm"] + BETA * agg["R"] + GAMMA * agg["W"]

    return agg.sort_values("biz_score", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO 4: Score Híbrido
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hybrid_pairs(
    df: pd.DataFrame,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    cust_to_idx: dict[str, int],
    doc_to_idx: dict[str, int],
    max_vol: int,
    max_lat: float,
) -> pd.DataFrame:
    """Combine MF score + Business score for customer→document pairs.

    Computes hybrid recommendation score by blending collaborative filtering
    predictions (MF) with business heuristics for all observed pairs.

    Args:
        df: Prepared DataFrame with access logs.
        user_factors: User latent factor matrix from SVD.
        item_factors: Item latent factor matrix from SVD.
        cust_to_idx: Customer identifier to matrix row index mapping.
        doc_to_idx: Document identifier to matrix column index mapping.
        max_vol: Maximum volume for normalization (typically len(df)).
        max_lat: Maximum latency for normalization (typically max httpTime).

    Returns:
        DataFrame sorted by hybrid score descending with columns:
        customerDocument, consultedDocument, volume, biz_score, mf_score, score.
    """
    log.info("Calculando score híbrido para pares...")

    # Score de negócio para pares
    biz = score_business(
        df, ["customerDocument", "consultedDocument"], max_vol, max_lat
    )

    # Score MF para pares
    mf_pairs = get_mf_scores_for_pairs(
        df, user_factors, item_factors, cust_to_idx, doc_to_idx
    )

    # Left-merge MF scores onto business scores
    merged = biz.merge(
        mf_pairs[["customerDocument", "consultedDocument", "mf_score"]],
        on=["customerDocument", "consultedDocument"],
        how="left",
    )
    merged["mf_score"] = merged["mf_score"].fillna(0.0)

    # Score híbrido: score_final = W_MF * mf_score + W_BIZ * biz_score
    merged["score"] = W_MF * merged["mf_score"] + W_BIZ * merged["biz_score"]

    return merged.sort_values("score", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO 5: Geração de Recomendações
# ═══════════════════════════════════════════════════════════════════════════════

def to_recs(
    df_dim: pd.DataFrame, cols: list[str], n: int, score_col: str = "biz_score"
) -> list[dict]:
    """Convert scored DataFrame to list of recommendation dicts.

    Args:
        df_dim: DataFrame with scored results (should already be sorted).
        cols: Identifier columns to include in each recommendation.
        n: Maximum number of recommendations to return.
        score_col: Column name containing the score to use.

    Returns:
        List of dicts with identifier columns (str), volume (int),
        and score (float rounded to 4 decimal places).
    """
    sorted_df = df_dim.sort_values(score_col, ascending=False)
    top_n = min(n, len(sorted_df))
    rows: list[dict] = []
    for r in sorted_df[cols + ["volume", score_col]].head(top_n).itertuples(index=False):
        rec: dict = {}
        for c in cols:
            rec[c] = str(getattr(r, c))
        rec["volume"] = int(getattr(r, "volume"))
        rec["score"] = round(float(getattr(r, score_col)), 4)
        rows.append(rec)
    return rows


def generate_recommendations(
    df: pd.DataFrame,
    hybrid_pairs: pd.DataFrame,
    t_ref: pd.Timestamp,
    model_metrics: dict,
) -> dict:
    """Assemble the complete output payload with all 5 recommendation dimensions.

    Generates top-N recommendations across: reports, features, customers,
    consulted documents (ranked by Business_Score), and pairs (ranked by
    Hybrid_Score). Assembles the full JSON payload including model metadata,
    parameters, metrics, and dataset statistics.

    Args:
        df: Prepared DataFrame with access logs.
        hybrid_pairs: DataFrame with hybrid-scored customer-document pairs.
        t_ref: Reference timestamp (max inclusionDate).
        model_metrics: Dict with model evaluation metrics.

    Returns:
        Complete recommendation payload dict ready for JSON serialization.
    """
    max_vol = len(df)
    max_lat = df["httpTime"].max()

    # Dimensões com score de negócio
    rep = score_business(df, ["reportName", "TYPE_REPORT"], max_vol, max_lat)
    feat = score_business(
        df, ["ID_FEATURE", "FEATURENAME", "FEATURE_TYPE"], max_vol, max_lat
    )
    cust = score_business(df, ["customerDocument"], max_vol, max_lat)
    cdoc = score_business(df, ["consultedDocument"], max_vol, max_lat)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "2.0-hybrid-svd",
        "model_params": {
            "formula": (
                "score_final = w_mf · score_mf + w_biz · "
                "(α·V_norm + β·exp(-λ·Δt) + γ·W_biz)"
            ),
            "W_biz_formula": (
                "W = 0.30·billing + 0.25·inquiry + 0.25·post_view"
                " + 0.20·latency_norm"
            ),
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "lambda": LAMBDA_DECAY,
            "w_mf": W_MF,
            "w_biz": W_BIZ,
            "svd_components": N_COMPONENTS,
            "svd_iterations": N_ITER,
        },
        "model_metrics": model_metrics,
        "warmup_targets": {
            "top_reports": to_recs(
                rep, ["reportName", "TYPE_REPORT"], TOP_N_REPORTS
            ),
            "top_features": to_recs(
                feat,
                ["ID_FEATURE", "FEATURENAME", "FEATURE_TYPE"],
                TOP_N_FEATURES,
            ),
            "top_customers": to_recs(
                cust, ["customerDocument"], TOP_N_CUSTOMERS
            ),
            "top_consulted_documents": to_recs(
                cdoc, ["consultedDocument"], TOP_N_CDOCS
            ),
            "top_pairs": to_recs(
                hybrid_pairs,
                ["customerDocument", "consultedDocument"],
                TOP_N_PAIRS,
                score_col="score",
            ),
        },
        "stats": {
            "total_records": len(df),
            "unique_customers": int(df["customerDocument"].nunique()),
            "unique_consulted": int(df["consultedDocument"].nunique()),
            "unique_reports": int(df["reportName"].nunique()),
            "unique_features": int(df["FEATURENAME"].nunique()),
            "date_range": {
                "from": str(df["inclusionDate"].min().date()),
                "to": str(t_ref.date()),
            },
        },
    }
    return payload


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO 6: Pipeline Principal
# ═══════════════════════════════════════════════════════════════════════════════

def run() -> str:
    """Execute the complete ML recommendation pipeline.

    Steps:
        1. Load and validate data
        2. Build interaction matrix
        3. Train SVD model
        4. Evaluate model quality
        5. Compute hybrid scores for pairs
        6. Generate recommendations across all dimensions
        7. Write output JSON
        8. Log summary and top-5 pairs

    Returns:
        Path to the output JSON file.
    """
    t_start = time.time()
    log.info("=" * 60)
    log.info("Cache Warmup Recommender v2.0 (ML-Enhanced)")
    log.info("=" * 60)

    # 1. Load and validate data
    df = load_data(INPUT_DIR)
    df, t_ref = prepare(df)
    max_vol = len(df)
    max_lat = df["httpTime"].max()

    # 2. Build interaction matrix
    matrix, cust_to_idx, doc_to_idx, customers, documents = (
        build_interaction_matrix(df)
    )

    # 3. Train SVD model (timed separately for model_metrics)
    t_train_start = time.time()
    svd, user_factors, item_factors, explained_var = train_svd(
        matrix, N_COMPONENTS, N_ITER
    )
    training_time = time.time() - t_train_start

    # 4. Evaluate model quality
    rmse = evaluate_model(matrix, svd, user_factors, item_factors)

    total_cells = matrix.shape[0] * matrix.shape[1]
    model_metrics = {
        "svd_explained_variance_pct": round(explained_var * 100, 2),
        "svd_reconstruction_rmse": round(rmse, 4),
        "matrix_shape": list(matrix.shape),
        "matrix_density_pct": round(
            (matrix.nnz / total_cells * 100) if total_cells > 0 else 0.0, 4
        ),
        "n_interactions": int(matrix.nnz),
        "training_time_seconds": round(training_time, 1),
    }

    # 5. Compute hybrid scores for pairs
    hybrid_pairs = compute_hybrid_pairs(
        df, user_factors, item_factors, cust_to_idx, doc_to_idx, max_vol, max_lat
    )

    # 6. Generate recommendations across all dimensions
    payload = generate_recommendations(df, hybrid_pairs, t_ref, model_metrics)

    # 7. Write output JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "warmup_recommendations.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    # 8. Log summary and top-5 pairs
    elapsed = time.time() - t_start
    log.info("Pipeline completo em %.1fs", elapsed)
    log.info("Output salvo em: %s", out_path)

    top_pairs = payload["warmup_targets"]["top_pairs"][:5]
    if top_pairs:
        log.info("Top-5 pares recomendados:")
        for p in top_pairs:
            log.info(
                "  customer=%s | doc=%s | score=%.4f | vol=%d",
                p["customerDocument"],
                p["consultedDocument"],
                p["score"],
                p["volume"],
            )

    return out_path


if __name__ == "__main__":
    run()
