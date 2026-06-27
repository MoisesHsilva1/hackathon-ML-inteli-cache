"""
lambda_trigger.py
Lambda 1 — Trigger: S3 Event → inicia SageMaker Processing Job
Evento: s3:ObjectCreated:* no bucket de input
"""
import boto3, json, os, logging
from datetime import datetime

log = logging.getLogger()
log.setLevel(logging.INFO)

REGION       = os.environ["AWS_REGION"]
ROLE_ARN     = os.environ["SAGEMAKER_ROLE_ARN"]
OUTPUT_BUCKET= os.environ["OUTPUT_BUCKET"]
OUTPUT_PREFIX= os.environ.get("OUTPUT_PREFIX", "output/warmup/")
CODE_S3_URI  = os.environ["CODE_S3_URI"]
IMAGE_URI    = os.environ.get(
    "IMAGE_URI",
    f"683313688378.dkr.ecr.{REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
)


def lambda_handler(event, context):
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    key    = record["s3"]["object"]["key"]

    if not key.endswith(".csv"):
        log.info("Arquivo ignorado (não é CSV): %s", key)
        return {"statusCode": 200, "body": "skipped"}

    job_name  = f"cache-warmup-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    s3_input  = f"s3://{bucket}/{key}"
    s3_output = f"s3://{OUTPUT_BUCKET}/{OUTPUT_PREFIX}"

    sm = boto3.client("sagemaker", region_name=REGION)
    sm.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={"ClusterConfig": {
            "InstanceCount": 1, "InstanceType": "ml.m5.xlarge", "VolumeSizeInGB": 20
        }},
        AppSpecification={
            "ImageUri": IMAGE_URI,
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/code/recommender.py"],
        },
        ProcessingInputs=[
            {"InputName": "input-data", "S3Input": {
                "S3Uri": s3_input, "LocalPath": "/opt/ml/processing/input",
                "S3DataType": "S3Prefix", "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            }},
            {"InputName": "code", "S3Input": {
                "S3Uri": CODE_S3_URI, "LocalPath": "/opt/ml/processing/code",
                "S3DataType": "S3Prefix", "S3InputMode": "File",
            }},
        ],
        ProcessingOutputConfig={"Outputs": [{"OutputName": "recommendations", "S3Output": {
            "S3Uri": s3_output, "LocalPath": "/opt/ml/processing/output", "S3UploadMode": "EndOfJob",
        }}]},
        Environment={
            "ALPHA": "0.35", "BETA": "0.40", "GAMMA": "0.25",
            "LAMBDA_DECAY": "0.15", "CSV_SEP": "|",
        },
        RoleArn=ROLE_ARN,
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
    )

    log.info("Job iniciado: %s | Input: %s | Output: %s", job_name, s3_input, s3_output)
    return {"statusCode": 200, "body": json.dumps({"job": job_name})}
