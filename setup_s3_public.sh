#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# setup_s3_public.sh — Habilita acesso público ao output JSON para o dashboard
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

BUCKET_NAME="${BUCKET_NAME:-s3-bucket-time-7}"
AWS_REGION="${AWS_REGION:-us-east-1}"

echo "▶ Configurando acesso público ao output do dashboard..."
echo "  Bucket: $BUCKET_NAME"
echo ""

# 1. Desabilitar Block Public Access para o bucket (apenas para o output)
echo "▶ [1/3] Removendo Block Public Access..."
aws s3api put-public-access-block \
    --bucket "$BUCKET_NAME" \
    --public-access-block-configuration \
    "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"
echo "  ✓ Block Public Access removido"

# 2. Aplicar bucket policy para permitir leitura pública do output/warmup/
echo ""
echo "▶ [2/3] Aplicando bucket policy (leitura pública em output/warmup/)..."

POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadWarmupOutput",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::${BUCKET_NAME}/output/warmup/*"
    }
  ]
}
EOF
)

aws s3api put-bucket-policy \
    --bucket "$BUCKET_NAME" \
    --policy "$POLICY"
echo "  ✓ Policy aplicada — output/warmup/* é público para leitura"

# 3. Configurar CORS para permitir fetch do GitHub Pages
echo ""
echo "▶ [3/3] Configurando CORS..."

CORS=$(cat <<EOF
{
  "CORSRules": [
    {
      "AllowedHeaders": ["*"],
      "AllowedMethods": ["GET"],
      "AllowedOrigins": [
        "https://moiseshsilva1.github.io",
        "http://localhost:*",
        "http://127.0.0.1:*"
      ],
      "ExposeHeaders": ["ETag"],
      "MaxAgeSeconds": 3600
    }
  ]
}
EOF
)

aws s3api put-bucket-cors \
    --bucket "$BUCKET_NAME" \
    --cors-configuration "$CORS"
echo "  ✓ CORS configurado para GitHub Pages + localhost"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ✅ Configuração concluída!"
echo ""
echo "  URL pública do JSON:"
echo "  https://${BUCKET_NAME}.s3.amazonaws.com/output/warmup/warmup_recommendations.json"
echo ""
echo "  O dashboard agora busca dados diretamente do S3."
echo "  Quando o SageMaker gerar um novo JSON, o dashboard atualiza automaticamente."
echo "══════════════════════════════════════════════════════════════"
