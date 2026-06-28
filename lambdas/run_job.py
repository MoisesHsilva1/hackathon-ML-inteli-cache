"""
run_job.py
Aciona o SageMaker Processing Job a partir de um evento S3
(chamado pela Lambda de trigger ou manualmente).
"""

import boto3, json, os
from datetime import datetime

# ── Configurações — ajuste para seu ambiente ──────────────────────────────────
REGION          = os.environ.get("AWS_REGION", "us-east-1")
ROLE_ARN        = os.environ.get("SAGEMAKER_ROLE_ARN",
                    "arn:aws:iam::719669559705:role/SageMakerExecutionRole")
INPUT_BUCKET    = os.environ.get("INPUT_BUCKET",  "s3-bucket-time-7")
INPUT_PREFIX    = os.environ.get("INPUT_PREFIX",  "input/")
OUTPUT_BUCKET   = os.environ.get("OUTPUT_BUCKET", "s3-bucket-time-7")
OUTPUT_PREFIX   = os.environ.get("OUTPUT_PREFIX", "output/warmup/")
CODE_S3_URI     = os.environ.get("CODE_S3_URI",
                    "s3://s3-bucket-time-7/code/recommender_ml.py")

# Imagem SKLearn gerenciada pela AWS (Python 3.10)
IMAGE_URI = (
    f"683313688378.dkr.ecr.{REGION}.amazonaws.com"
    "/sagemaker-scikit-learn:1.2-1-cpu-py3"
)

# ── Parâmetros do modelo (todos opcionais — têm defaults no script) ───────────
MODEL_PARAMS = {
    "ALPHA":         "0.35",
    "BETA":          "0.40",
    "GAMMA":         "0.25",
    "LAMBDA_DECAY":  "0.15",
    "W_MF":          "0.45",
    "W_BIZ":         "0.55",
    "N_COMPONENTS":  "50",
    "N_ITER":        "15",
    "TOP_N_REPORTS": "20",
    "TOP_N_FEATURES":"30",
    "TOP_N_CUSTOMERS":"500",
    "TOP_N_CDOCS":   "1000",
    "TOP_N_PAIRS":   "2000",
    "CSV_SEP":       "|",
}


def start_job(input_s3_uri: str | None = None) -> str:
    """Dispara o Processing Job e retorna o nome do job."""
    sm = boto3.client("sagemaker", region_name=REGION)

    job_name = f"cache-warmup-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    s3_input  = input_s3_uri or f"s3://{INPUT_BUCKET}/{INPUT_PREFIX}"
    s3_output = f"s3://{OUTPUT_BUCKET}/{OUTPUT_PREFIX}"

    response = sm.create_processing_job(
        ProcessingJobName=job_name,

        # Recurso de computação
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType":  "ml.m5.xlarge",   # 4 vCPU, 16 GB — suficiente para ~1M registros
                "VolumeSizeInGB": 20,
            }
        },

        # Imagem e código
        AppSpecification={
            "ImageUri": IMAGE_URI,
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/code/recommender_ml.py"],
        },

        # Canal de entrada: CSV do S3
        ProcessingInputs=[
            {
                "InputName": "input-data",
                "S3Input": {
                    "S3Uri":           s3_input,
                    "LocalPath":       "/opt/ml/processing/input",
                    "S3DataType":      "S3Prefix",
                    "S3InputMode":     "File",
                    "S3DataDistributionType": "FullyReplicated",
                },
            },
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri":       CODE_S3_URI,
                    "LocalPath":   "/opt/ml/processing/code",
                    "S3DataType":  "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ],

        # Canal de saída: JSON → S3
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "recommendations",
                    "S3Output": {
                        "S3Uri":        s3_output,
                        "LocalPath":    "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },

        # Variáveis de ambiente (parâmetros do modelo)
        Environment=MODEL_PARAMS,

        # IAM Role com permissões em S3 + SageMaker
        RoleArn=ROLE_ARN,

        # Tempo máximo de execução
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
    )

    print(f"✅ Job iniciado: {job_name}")
    print(f"   Input : {s3_input}")
    print(f"   Output: {s3_output}")
    return job_name


# ── Handler para Lambda (trigger S3) ─────────────────────────────────────────
def lambda_handler(event, context):
    """
    Invocada pelo S3 Event Notification quando um CSV é depositado no bucket.
    """
    record     = event["Records"][0]
    bucket     = record["s3"]["bucket"]["name"]
    key        = record["s3"]["object"]["key"]
    s3_input   = f"s3://{bucket}/{key}"

    job_name = start_job(s3_input)
    return {"statusCode": 200, "body": json.dumps({"job": job_name})}


if __name__ == "__main__":
    start_job()
