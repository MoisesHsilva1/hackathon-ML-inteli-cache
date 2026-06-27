"""
cache_warmup_recommender.py
SageMaker Processing Job — Cache Warmup Recommender
Entrada : s3://<INPUT_BUCKET>/<INPUT_PREFIX>/data.csv
Saída   : s3://<OUTPUT_BUCKET>/<OUTPUT_PREFIX>/warmup_recommendations.json
"""

import json, os, logging
import boto3, numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Parâmetros via env vars (injetados pelo SageMaker) ───────────────────────
ALPHA        = float(os.environ.get("ALPHA",        0.35))
BETA         = float(os.environ.get("BETA",         0.40))
GAMMA        = float(os.environ.get("GAMMA",        0.25))
LAMBDA_DECAY = float(os.environ.get("LAMBDA_DECAY", 0.15))

TOP_N_REPORTS   = int(os.environ.get("TOP_N_REPORTS",   20))
TOP_N_FEATURES  = int(os.environ.get("TOP_N_FEATURES",  30))
TOP_N_CUSTOMERS = int(os.environ.get("TOP_N_CUSTOMERS", 500))
TOP_N_CDOCS     = int(os.environ.get("TOP_N_CDOCS",    1000))
TOP_N_PAIRS     = int(os.environ.get("TOP_N_PAIRS",    2000))
TOP_N_PAIRS_CUST= int(os.environ.get("TOP_N_PAIRS_CUST", 200))

# SageMaker monta os canais em /opt/ml/processing/input e /opt/ml/processing/output
INPUT_DIR  = os.environ.get("SM_INPUT_DIR",  "/opt/ml/processing/input")
OUTPUT_DIR = os.environ.get("SM_OUTPUT_DIR", "/opt/ml/processing/output")
CSV_SEP    = os.environ.get("CSV_SEP", "|")

AGG_SPEC = dict(
    volume       =("ID_REPORT",        "count"),
    delta_mean   =("delta_days",       "mean"),
    billing_mean =("billing",          "mean"),
    inquiry_mean =("inquiry",          "mean"),
    view_mean    =("post_report_view", "mean"),
    httpTime_mean=("httpTime",         "mean"),
)

def prepare(df):
    df = df.copy()
    df["inclusionDate"]    = pd.to_datetime(df["inclusionDate"], utc=False)
    df["billing"]          = df["billing"].astype(bool)
    df["inquiry"]          = df["inquiry"].astype(bool)
    df["post_report_view"] = df["post_report_view"].astype(bool)
    t_ref = df["inclusionDate"].max()
    df["delta_days"] = (t_ref - df["inclusionDate"]).dt.total_seconds() / 86400.0
    return df, t_ref

def score_dim(df, group_cols, max_vol, max_lat):
    agg = df.groupby(group_cols).agg(**AGG_SPEC).reset_index()
    agg["V_norm"] = agg["volume"] / max_vol
    agg["R"]      = np.exp(-LAMBDA_DECAY * agg["delta_mean"])
    agg["W"]      = (0.30*agg["billing_mean"] + 0.25*agg["inquiry_mean"]
                   + 0.25*agg["view_mean"]    + 0.20*(agg["httpTime_mean"]/max_lat))
    agg["score"]  = ALPHA*agg["V_norm"] + BETA*agg["R"] + GAMMA*agg["W"]
    return agg.sort_values("score", ascending=False)

def to_recs(d, cols, n):
    rows = []
    for r in d[cols + ["volume","score"]].head(n).itertuples(index=False):
        rec = {c: getattr(r, c) for c in cols + ["volume","score"]}
        rec["score"]  = round(float(rec["score"]), 4)
        rec["volume"] = int(rec["volume"])
        for k in ["ID_FEATURE","customerDocument","consultedDocument"]:
            if k in rec: rec[k] = str(rec[k])
        rows.append(rec)
    return rows

def run():
    # 1. Localiza CSV no canal de entrada
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV em {INPUT_DIR}")
    csv_path = os.path.join(INPUT_DIR, csv_files[0])
    log.info("Lendo %s", csv_path)

    df = pd.read_csv(csv_path, sep=CSV_SEP)
    df, t_ref = prepare(df)
    max_vol = len(df)
    max_lat = df["httpTime"].max()
    log.info("Registros: %d | Janela: %s → %s", len(df),
             df["inclusionDate"].min().date(), t_ref.date())

    # 2. Score por dimensão
    rep  = score_dim(df, ["reportName","TYPE_REPORT"],              max_vol, max_lat)
    feat = score_dim(df, ["ID_FEATURE","FEATURENAME","FEATURE_TYPE"], max_vol, max_lat)
    cust = score_dim(df, ["customerDocument"],                      max_vol, max_lat)
    cdoc = score_dim(df, ["consultedDocument"],                     max_vol, max_lat)

    top_cust_ids = set(cust.head(TOP_N_PAIRS_CUST)["customerDocument"].tolist())
    df_sub = df[df["customerDocument"].isin(top_cust_ids)]
    pairs_max_vol = df_sub.groupby(["customerDocument","consultedDocument"]).size().max()
    pairs = score_dim(df_sub, ["customerDocument","consultedDocument"], pairs_max_vol, max_lat)

    # 3. Monta payload
    payload = {
        "generated_at": t_ref.isoformat(),
        "model_params": {
            "formula":        "score = α·V_norm + β·exp(-λ·Δt) + γ·W_biz",
            "W_biz_formula":  "W = 0.30·billing + 0.25·inquiry + 0.25·post_view + 0.20·latency_norm",
            "alpha":  ALPHA, "beta": BETA, "gamma": GAMMA, "lambda": LAMBDA_DECAY,
        },
        "warmup_targets": {
            "top_reports":             to_recs(rep,   ["reportName","TYPE_REPORT"],                     TOP_N_REPORTS),
            "top_features":            to_recs(feat,  ["ID_FEATURE","FEATURENAME","FEATURE_TYPE"],      TOP_N_FEATURES),
            "top_customers":           to_recs(cust,  ["customerDocument"],                             TOP_N_CUSTOMERS),
            "top_consulted_documents": to_recs(cdoc,  ["consultedDocument"],                            TOP_N_CDOCS),
            "top_pairs":               to_recs(pairs, ["customerDocument","consultedDocument"],         TOP_N_PAIRS),
        },
        "stats": {
            "total_records":    len(df),
            "unique_customers": int(df["customerDocument"].nunique()),
            "unique_consulted": int(df["consultedDocument"].nunique()),
            "unique_reports":   int(df["reportName"].nunique()),
            "unique_features":  int(df["FEATURENAME"].nunique()),
            "date_range": {"from": str(df["inclusionDate"].min().date()), "to": str(t_ref.date())},
        },
    }

    # 4. Escreve no canal de saída (SageMaker faz o upload para S3 automaticamente)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "warmup_recommendations.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    log.info("✅ Salvo em %s", out_path)

if __name__ == "__main__":
    run()
