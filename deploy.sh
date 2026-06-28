#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# deploy.sh — Deploy completo do Cache Warmup Recommender na AWS
# Provisiona: S3, IAM Roles, Lambda (trigger + warmup + run_job), e faz upload do código
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Configuração ─────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET_NAME="${BUCKET_NAME:-s3-bucket-time-7}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Nomes dos recursos
SAGEMAKER_ROLE_NAME="SageMakerExecutionRole"
LAMBDA_ROLE_NAME="CacheWarmupLambdaRole"
LAMBDA_TRIGGER_NAME="cache-warmup-trigger"
LAMBDA_WARMUP_NAME="cache-warmup-warmup"
LAMBDA_RUNJOB_NAME="cache-warmup-run-job"

# Variáveis para as Lambdas
API_BASE_URL="${API_BASE_URL:-https://api.example.com/v1}"
API_KEY="${API_KEY:-CHANGE_ME}"

echo "══════════════════════════════════════════════════════════════"
echo "  Cache Warmup Recommender — Deploy AWS"
echo "══════════════════════════════════════════════════════════════"
echo "  Região:     $AWS_REGION"
echo "  Conta:      $ACCOUNT_ID"
echo "  Bucket:     $BUCKET_NAME"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── 1. Criar bucket S3 ──────────────────────────────────────────────────────
echo "▶ [1/7] Criando bucket S3..."

if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    echo "  ✓ Bucket '$BUCKET_NAME' já existe"
else
    if [ "$AWS_REGION" = "us-east-1" ]; then
        aws s3api create-bucket \
            --bucket "$BUCKET_NAME" \
            --region "$AWS_REGION"
    else
        aws s3api create-bucket \
            --bucket "$BUCKET_NAME" \
            --region "$AWS_REGION" \
            --create-bucket-configuration LocationConstraint="$AWS_REGION"
    fi
    echo "  ✓ Bucket '$BUCKET_NAME' criado"
fi

# Criar prefixos (objetos vazios para simular pastas)
aws s3api put-object --bucket "$BUCKET_NAME" --key "input/" --content-length 0 2>/dev/null || true
aws s3api put-object --bucket "$BUCKET_NAME" --key "output/warmup/" --content-length 0 2>/dev/null || true
aws s3api put-object --bucket "$BUCKET_NAME" --key "code/" --content-length 0 2>/dev/null || true
echo "  ✓ Prefixos criados (input/, output/warmup/, code/)"

# ── 2. Criar IAM Role para SageMaker ────────────────────────────────────────
echo ""
echo "▶ [2/7] Criando IAM Role para SageMaker..."

SAGEMAKER_TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "sagemaker.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

if aws iam get-role --role-name "$SAGEMAKER_ROLE_NAME" 2>/dev/null; then
    echo "  ✓ Role '$SAGEMAKER_ROLE_NAME' já existe"
else
    aws iam create-role \
        --role-name "$SAGEMAKER_ROLE_NAME" \
        --assume-role-policy-document "$SAGEMAKER_TRUST_POLICY" \
        --description "Role para SageMaker Processing Job - Cache Warmup Recommender"
    echo "  ✓ Role '$SAGEMAKER_ROLE_NAME' criada"
fi

# Atualiza a policy inline do SageMaker
SAGEMAKER_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3InputOutput",
      "Effect": "Allow",
      "Action": ["s3:GetObject","s3:PutObject","s3:ListBucket","s3:DeleteObject"],
      "Resource": [
        "arn:aws:s3:::${BUCKET_NAME}",
        "arn:aws:s3:::${BUCKET_NAME}/*"
      ]
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],
      "Resource": "arn:aws:logs:*:${ACCOUNT_ID}:log-group:/aws/sagemaker/ProcessingJobs*"
    },
    {
      "Sid": "ECRPull",
      "Effect": "Allow",
      "Action": ["ecr:GetDownloadUrlForLayer","ecr:BatchGetImage","ecr:GetAuthorizationToken"],
      "Resource": "*"
    }
  ]
}
EOF
)

aws iam put-role-policy \
    --role-name "$SAGEMAKER_ROLE_NAME" \
    --policy-name "CacheWarmupSageMakerPolicy" \
    --policy-document "$SAGEMAKER_POLICY"
echo "  ✓ Policy inline aplicada ao SageMaker Role"

SAGEMAKER_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${SAGEMAKER_ROLE_NAME}"

# ── 3. Criar IAM Role para Lambda ───────────────────────────────────────────
echo ""
echo "▶ [3/7] Criando IAM Role para Lambda..."

LAMBDA_TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

if aws iam get-role --role-name "$LAMBDA_ROLE_NAME" 2>/dev/null; then
    echo "  ✓ Role '$LAMBDA_ROLE_NAME' já existe"
else
    aws iam create-role \
        --role-name "$LAMBDA_ROLE_NAME" \
        --assume-role-policy-document "$LAMBDA_TRUST_POLICY" \
        --description "Role para Lambdas do Cache Warmup Recommender"
    echo "  ✓ Role '$LAMBDA_ROLE_NAME' criada"
fi

LAMBDA_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "StartSageMakerJob",
      "Effect": "Allow",
      "Action": ["sagemaker:CreateProcessingJob","sagemaker:DescribeProcessingJob"],
      "Resource": "*"
    },
    {
      "Sid": "PassRoleToSageMaker",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "${SAGEMAKER_ROLE_ARN}"
    },
    {
      "Sid": "S3Access",
      "Effect": "Allow",
      "Action": ["s3:GetObject","s3:PutObject","s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::${BUCKET_NAME}",
        "arn:aws:s3:::${BUCKET_NAME}/*"
      ]
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],
      "Resource": "arn:aws:logs:*:${ACCOUNT_ID}:*"
    }
  ]
}
EOF
)

aws iam put-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-name "CacheWarmupLambdaPolicy" \
    --policy-document "$LAMBDA_POLICY"
echo "  ✓ Policy inline aplicada ao Lambda Role"

LAMBDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${LAMBDA_ROLE_NAME}"

# Aguarda propagação do IAM (necessário para Lambda)
echo "  ⏳ Aguardando propagação IAM (10s)..."
sleep 10

# ── 4. Upload do código para S3 ─────────────────────────────────────────────
echo ""
echo "▶ [4/7] Fazendo upload do código para S3..."

aws s3 cp "$PROJECT_DIR/src/recommender_ml.py" "s3://${BUCKET_NAME}/code/recommender_ml.py"
echo "  ✓ recommender_ml.py → s3://${BUCKET_NAME}/code/"

# ── 5. Deploy Lambda Trigger ────────────────────────────────────────────────
echo ""
echo "▶ [5/7] Deploy da Lambda Trigger (S3 CSV → SageMaker Job)..."

cd "$PROJECT_DIR/lambdas"
zip -j /tmp/trigger.zip trigger.py

if aws lambda get-function --function-name "$LAMBDA_TRIGGER_NAME" 2>/dev/null; then
    aws lambda update-function-code \
        --function-name "$LAMBDA_TRIGGER_NAME" \
        --zip-file fileb:///tmp/trigger.zip \
        --region "$AWS_REGION" > /dev/null

    aws lambda update-function-configuration \
        --function-name "$LAMBDA_TRIGGER_NAME" \
        --environment "Variables={SAGEMAKER_ROLE_ARN=${SAGEMAKER_ROLE_ARN},OUTPUT_BUCKET=${BUCKET_NAME},OUTPUT_PREFIX=output/warmup/,CODE_S3_URI=s3://${BUCKET_NAME}/code/}" \
        --region "$AWS_REGION" > /dev/null
    echo "  ✓ Lambda '$LAMBDA_TRIGGER_NAME' atualizada"
else
    aws lambda create-function \
        --function-name "$LAMBDA_TRIGGER_NAME" \
        --runtime python3.12 \
        --role "$LAMBDA_ROLE_ARN" \
        --handler "trigger.lambda_handler" \
        --zip-file fileb:///tmp/trigger.zip \
        --timeout 60 \
        --memory-size 128 \
        --environment "Variables={SAGEMAKER_ROLE_ARN=${SAGEMAKER_ROLE_ARN},OUTPUT_BUCKET=${BUCKET_NAME},OUTPUT_PREFIX=output/warmup/,CODE_S3_URI=s3://${BUCKET_NAME}/code/}" \
        --region "$AWS_REGION" > /dev/null
    echo "  ✓ Lambda '$LAMBDA_TRIGGER_NAME' criada"
fi

# ── 6. Deploy Lambda Warmup ─────────────────────────────────────────────────
echo ""
echo "▶ [6/7] Deploy da Lambda Warmup (JSON → API calls)..."

zip -j /tmp/warmup.zip warmup.py

if aws lambda get-function --function-name "$LAMBDA_WARMUP_NAME" 2>/dev/null; then
    aws lambda update-function-code \
        --function-name "$LAMBDA_WARMUP_NAME" \
        --zip-file fileb:///tmp/warmup.zip \
        --region "$AWS_REGION" > /dev/null

    aws lambda update-function-configuration \
        --function-name "$LAMBDA_WARMUP_NAME" \
        --environment "Variables={API_BASE_URL=${API_BASE_URL},API_KEY=${API_KEY},MAX_PAIRS=500,TIMEOUT_SECONDS=10}" \
        --region "$AWS_REGION" > /dev/null
    echo "  ✓ Lambda '$LAMBDA_WARMUP_NAME' atualizada"
else
    aws lambda create-function \
        --function-name "$LAMBDA_WARMUP_NAME" \
        --runtime python3.12 \
        --role "$LAMBDA_ROLE_ARN" \
        --handler "warmup.lambda_handler" \
        --zip-file fileb:///tmp/warmup.zip \
        --timeout 300 \
        --memory-size 256 \
        --environment "Variables={API_BASE_URL=${API_BASE_URL},API_KEY=${API_KEY},MAX_PAIRS=500,TIMEOUT_SECONDS=10}" \
        --region "$AWS_REGION" > /dev/null
    echo "  ✓ Lambda '$LAMBDA_WARMUP_NAME' criada"
fi

# ── 7. Configurar S3 Event Notifications ────────────────────────────────────
echo ""
echo "▶ [7/7] Configurando S3 Event Notifications..."

# Permissão para o S3 invocar a Lambda Trigger
TRIGGER_ARN=$(aws lambda get-function --function-name "$LAMBDA_TRIGGER_NAME" --query 'Configuration.FunctionArn' --output text)

aws lambda add-permission \
    --function-name "$LAMBDA_TRIGGER_NAME" \
    --statement-id "S3InvokeTrigger" \
    --action "lambda:InvokeFunction" \
    --principal s3.amazonaws.com \
    --source-arn "arn:aws:s3:::${BUCKET_NAME}" \
    --source-account "$ACCOUNT_ID" 2>/dev/null || true

# Permissão para o S3 invocar a Lambda Warmup
WARMUP_ARN=$(aws lambda get-function --function-name "$LAMBDA_WARMUP_NAME" --query 'Configuration.FunctionArn' --output text)

aws lambda add-permission \
    --function-name "$LAMBDA_WARMUP_NAME" \
    --statement-id "S3InvokeWarmup" \
    --action "lambda:InvokeFunction" \
    --principal s3.amazonaws.com \
    --source-arn "arn:aws:s3:::${BUCKET_NAME}" \
    --source-account "$ACCOUNT_ID" 2>/dev/null || true

# Configura as notificações do S3
NOTIFICATION_CONFIG=$(cat <<EOF
{
  "LambdaFunctionConfigurations": [
    {
      "Id": "TriggerOnCSVUpload",
      "LambdaFunctionArn": "${TRIGGER_ARN}",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            {"Name": "prefix", "Value": "input/"},
            {"Name": "suffix", "Value": ".csv"}
          ]
        }
      }
    },
    {
      "Id": "WarmupOnRecommendationJSON",
      "LambdaFunctionArn": "${WARMUP_ARN}",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            {"Name": "prefix", "Value": "output/warmup/"},
            {"Name": "suffix", "Value": ".json"}
          ]
        }
      }
    }
  ]
}
EOF
)

aws s3api put-bucket-notification-configuration \
    --bucket "$BUCKET_NAME" \
    --notification-configuration "$NOTIFICATION_CONFIG"
echo "  ✓ S3 Event Notifications configuradas"
echo "    • input/*.csv → Lambda Trigger → SageMaker Job"
echo "    • output/warmup/*.json → Lambda Warmup → API calls"

# ── Resumo ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ✅ Deploy concluído com sucesso!"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  Recursos criados:"
echo "    • S3 Bucket:       $BUCKET_NAME"
echo "    • IAM Role SM:     $SAGEMAKER_ROLE_ARN"
echo "    • IAM Role Lambda: $LAMBDA_ROLE_ARN"
echo "    • Lambda Trigger:  $LAMBDA_TRIGGER_NAME"
echo "    • Lambda Warmup:   $LAMBDA_WARMUP_NAME"
echo ""
echo "  Fluxo automático:"
echo "    1. Upload CSV → s3://$BUCKET_NAME/input/dados.csv"
echo "    2. Lambda Trigger detecta e inicia SageMaker Processing Job"
echo "    3. SageMaker gera recomendações → s3://$BUCKET_NAME/output/warmup/"
echo "    4. Lambda Warmup detecta JSON e chama APIs para pré-aquecer cache"
echo ""
echo "  Para testar:"
echo "    aws s3 cp data/sample_input.csv s3://$BUCKET_NAME/input/dados.csv"
echo ""
echo "  Para monitorar:"
echo "    aws sagemaker list-processing-jobs --sort-by CreationTime --sort-order Descending --max-results 5"
echo "    aws logs tail /aws/lambda/$LAMBDA_TRIGGER_NAME --follow"
echo "    aws logs tail /aws/lambda/$LAMBDA_WARMUP_NAME --follow"
echo ""

# Limpeza de arquivos temporários
rm -f /tmp/trigger.zip /tmp/warmup.zip
