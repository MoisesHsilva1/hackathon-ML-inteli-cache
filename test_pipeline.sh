#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# test_pipeline.sh — Testa o pipeline completo: upload CSV → SageMaker → Output
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
BUCKET_NAME="${BUCKET_NAME:-s3-bucket-time-7}"
INPUT_FILE="${1:-data/sample_input.csv}"
POLL_INTERVAL=30

echo "══════════════════════════════════════════════════════════════"
echo "   Teste do Pipeline — Cache Warmup Recommender"
echo "══════════════════════════════════════════════════════════════"
echo "  Bucket:     $BUCKET_NAME"
echo "  Input:      $INPUT_FILE"
echo "  Região:     $AWS_REGION"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── 1. Verificar se o arquivo existe ─────────────────────────────────────────
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Arquivo não encontrado: $INPUT_FILE"
    echo "   Use: ./test_pipeline.sh <caminho-do-csv>"
    exit 1
fi

FILE_SIZE=$(wc -c < "$INPUT_FILE" | tr -d ' ')
LINE_COUNT=$(wc -l < "$INPUT_FILE" | tr -d ' ')
echo "📄 Arquivo: $INPUT_FILE ($LINE_COUNT linhas, $(numfmt --to=iec $FILE_SIZE 2>/dev/null || echo "${FILE_SIZE}B"))"
echo ""

# ── 2. Upload do CSV para S3 (dispara o pipeline) ───────────────────────────
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
S3_KEY="input/test_${TIMESTAMP}.csv"

echo "▶ [1/4] Fazendo upload para s3://$BUCKET_NAME/$S3_KEY..."
aws s3 cp "$INPUT_FILE" "s3://$BUCKET_NAME/$S3_KEY" --region "$AWS_REGION"
echo "  ✓ Upload concluído"
echo ""

# ── 3. Aguardar o Processing Job iniciar ─────────────────────────────────────
echo "▶ [2/4] Aguardando SageMaker Processing Job iniciar..."
echo "  (A Lambda trigger deve detectar o upload e criar o job)"
echo ""

sleep 15  # Dá tempo para a Lambda executar

# Busca o job mais recente
JOB_NAME=$(aws sagemaker list-processing-jobs \
    --sort-by CreationTime \
    --sort-order Descending \
    --max-results 1 \
    --region "$AWS_REGION" \
    --query 'ProcessingJobSummaries[0].ProcessingJobName' \
    --output text 2>/dev/null || echo "NONE")

if [ "$JOB_NAME" = "NONE" ] || [ "$JOB_NAME" = "None" ]; then
    echo "  ⚠️  Nenhum job encontrado. Verificando logs da Lambda..."
    aws logs tail "/aws/lambda/cache-warmup-trigger" \
        --since 2m --region "$AWS_REGION" 2>/dev/null | tail -5 || true
    echo ""
    echo "  💡 Pode ser que a Lambda ainda não foi invocada. Aguarde mais alguns segundos."
    sleep 20
    JOB_NAME=$(aws sagemaker list-processing-jobs \
        --sort-by CreationTime --sort-order Descending --max-results 1 \
        --region "$AWS_REGION" \
        --query 'ProcessingJobSummaries[0].ProcessingJobName' --output text)
fi

echo "  📋 Job encontrado: $JOB_NAME"
echo ""

# ── 4. Monitorar o job até completar ─────────────────────────────────────────
echo "▶ [3/4] Monitorando execução do job..."
echo "  (Polling a cada ${POLL_INTERVAL}s — ml.m5.xlarge geralmente leva 3-10min)"
echo ""

while true; do
    STATUS=$(aws sagemaker describe-processing-job \
        --processing-job-name "$JOB_NAME" \
        --region "$AWS_REGION" \
        --query 'ProcessingJobStatus' \
        --output text)

    case "$STATUS" in
        InProgress)
            echo "  ⏳ $(date +%H:%M:%S) — Status: $STATUS"
            sleep $POLL_INTERVAL
            ;;
        Completed)
            echo "  ✅ $(date +%H:%M:%S) — Job CONCLUÍDO!"
            break
            ;;
        Failed|Stopped)
            echo "  ❌ $(date +%H:%M:%S) — Job $STATUS!"
            echo ""
            echo "  Detalhes do erro:"
            aws sagemaker describe-processing-job \
                --processing-job-name "$JOB_NAME" \
                --region "$AWS_REGION" \
                --query '{Status: ProcessingJobStatus, Reason: FailureReason, ExitMessage: ExitMessage}' \
                --output table
            exit 1
            ;;
        *)
            echo "  ⏳ $(date +%H:%M:%S) — Status: $STATUS"
            sleep $POLL_INTERVAL
            ;;
    esac
done

echo ""

# ── 5. Baixar e verificar output ─────────────────────────────────────────────
echo "▶ [4/4] Verificando output..."

OUTPUT_KEY="output/warmup/warmup_recommendations.json"
LOCAL_OUTPUT="dashboard/data.json"

if aws s3 cp "s3://$BUCKET_NAME/$OUTPUT_KEY" "$LOCAL_OUTPUT" --region "$AWS_REGION" 2>/dev/null; then
    echo "  ✓ Output baixado: $LOCAL_OUTPUT"
else
    # Tenta buscar qualquer JSON no output
    echo "  ⚠️  Arquivo padrão não encontrado, buscando alternativas..."
    FOUND=$(aws s3 ls "s3://$BUCKET_NAME/output/warmup/" --region "$AWS_REGION" | grep ".json" | tail -1 | awk '{print $4}')
    if [ -n "$FOUND" ]; then
        aws s3 cp "s3://$BUCKET_NAME/output/warmup/$FOUND" "$LOCAL_OUTPUT" --region "$AWS_REGION"
        echo "  ✓ Output encontrado e baixado: $FOUND → $LOCAL_OUTPUT"
    else
        echo "  ❌ Nenhum JSON encontrado no output"
        exit 1
    fi
fi

echo ""

# ── 6. Resumo dos resultados ─────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  ✅ Pipeline Executado com Sucesso!"
echo "══════════════════════════════════════════════════════════════"
echo ""

# Parse JSON com python (disponível em qualquer máquina)
python3 -c "
import json, sys

with open('$LOCAL_OUTPUT') as f:
    d = json.load(f)

mm = d.get('model_metrics', {})
stats = d.get('stats', {})
targets = d.get('warmup_targets', {})

print('  📊 Métricas do Modelo:')
if mm:
    print(f'     Variância Explicada:  {mm.get(\"svd_explained_variance_pct\", \"N/A\")}%')
    print(f'     RMSE Reconstrução:    {mm.get(\"svd_reconstruction_rmse\", \"N/A\")}')
    print(f'     Densidade Matriz:     {mm.get(\"matrix_density_pct\", \"N/A\")}%')
    print(f'     Tempo Treinamento:    {mm.get(\"training_time_seconds\", \"N/A\")}s')
else:
    print('     (modelo v1 — sem métricas ML)')

print()
print('  📋 Recomendações Geradas:')
print(f'     Top Reports:      {len(targets.get(\"top_reports\", []))}')
print(f'     Top Features:     {len(targets.get(\"top_features\", []))}')
print(f'     Top Customers:    {len(targets.get(\"top_customers\", []))}')
print(f'     Top Pairs:        {len(targets.get(\"top_pairs\", []))}')

print()
print('  📈 Dataset:')
print(f'     Registros:        {stats.get(\"total_records\", \"?\"):,}')
print(f'     Customers:        {stats.get(\"unique_customers\", \"?\"):,}')
print(f'     Período:          {stats.get(\"date_range\", {}).get(\"from\", \"?\")} → {stats.get(\"date_range\", {}).get(\"to\", \"?\")}')

top5 = targets.get('top_pairs', [])[:5]
if top5:
    print()
    print('  🔥 Top-5 Pares Recomendados:')
    for i, p in enumerate(top5, 1):
        print(f'     {i}. customer={p[\"customerDocument\"]} doc={p[\"consultedDocument\"]} score={p[\"score\"]}')
"

echo ""
echo "  🌐 Dashboard: abra dashboard/index.html no browser"
echo "  📁 Dados: $LOCAL_OUTPUT"
echo ""
