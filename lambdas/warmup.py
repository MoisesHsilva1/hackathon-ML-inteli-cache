"""
lambda_warmup.py
Lambda 2 — Warmup: lê o JSON de recomendações do S3 e chama as APIs
Evento: s3:ObjectCreated:* no bucket de output (warmup_recommendations.json)
"""
import boto3, json, os, logging, urllib.request, urllib.error

log = logging.getLogger()
log.setLevel(logging.INFO)

API_BASE_URL    = os.environ["API_BASE_URL"]       # ex: https://api.experian.com.br/v1
API_KEY         = os.environ["API_KEY"]
MAX_PAIRS       = int(os.environ.get("MAX_PAIRS", "500"))   # limita chamadas por execução
TIMEOUT_SECONDS = int(os.environ.get("TIMEOUT_SECONDS", "10"))


def call_api(customer_document: str, consulted_document: str, report_name: str) -> int:
    """Faz a chamada de warmup à API. Retorna o HTTP status code."""
    url = f"{API_BASE_URL}/reports/{report_name}?customer={customer_document}&consulted={consulted_document}"
    req = urllib.request.Request(url, headers={"x-api-key": API_KEY, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as exc:
        log.warning("Erro na chamada: %s", exc)
        return 0


def lambda_handler(event, context):
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    key    = record["s3"]["object"]["key"]

    if "warmup_recommendations.json" not in key:
        return {"statusCode": 200, "body": "skipped"}

    # Lê o JSON de recomendações
    s3   = boto3.client("s3")
    obj  = s3.get_object(Bucket=bucket, Key=key)
    data = json.loads(obj["Body"].read())

    pairs    = data["warmup_targets"]["top_pairs"][:MAX_PAIRS]
    reports  = {r["reportName"]: r for r in data["warmup_targets"]["top_reports"]}

    ok = err = 0
    for pair in pairs:
        cust   = pair["customerDocument"]
        cdoc   = pair["consultedDocument"]
        # Usa o relatório de maior score para esse par (fallback: primeiro)
        report = list(reports.keys())[0] if reports else "RELATORIO_AVANCADO_TOP_SCORE_PJ"

        status = call_api(cust, cdoc, report)
        if 200 <= status < 300:
            ok += 1
        else:
            err += 1

    log.info("Warmup concluído: %d ok / %d erros | gerado_em: %s",
             ok, err, data.get("generated_at","?"))
    return {"statusCode": 200, "body": json.dumps({"ok": ok, "errors": err})}
