# Tech Stack

## Language & Runtime

- Python 3.10+ (SageMaker Processing Job)
- Python 3.12 (AWS Lambda functions)

## AWS Services

| Service | Purpose |
|---------|---------|
| SageMaker Processing Job | Runs the recommendation model (ml.m5.xlarge) |
| S3 | Input CSVs, output JSON, code storage |
| Lambda | Event-driven triggers and warmup execution |
| IAM | Role-based access for SageMaker and Lambda |
| EventBridge | Optional scheduled execution (daily cron) |
| CloudWatch | Logging and monitoring |

## Python Libraries

- `pandas` — data manipulation and aggregation
- `numpy` — exponential decay calculations
- `boto3` — AWS SDK (S3, SageMaker clients)
- `urllib.request` — HTTP calls in warmup Lambda (no external dependencies)

## Container Image

- AWS-managed SageMaker scikit-learn image: `683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3`

## Key Configuration

- Model parameters injected via environment variables (`ALPHA`, `BETA`, `GAMMA`, `LAMBDA_DECAY`, `TOP_N_*`)
- CSV separator: pipe `|`
- SageMaker channels: `/opt/ml/processing/input` and `/opt/ml/processing/output`

## Common Commands

```bash
# Upload recommender script to S3
aws s3 cp src/recommender.py s3://<BUCKET>/code/recommender.py

# Upload input CSV
aws s3 cp data/sample_input.csv s3://<BUCKET>/input/data.csv

# Run processing job manually
python lambdas/run_job.py

# Check job status
aws sagemaker describe-processing-job \
  --processing-job-name <job-name> \
  --query "ProcessingJobStatus"

# Inspect output
aws s3 cp s3://<BUCKET>/output/warmup/warmup_recommendations.json - | python3 -m json.tool | head -50
```

## No Build System

This is a collection of standalone Python scripts — no package manager, no build step, no test framework currently configured. Scripts are deployed individually to S3 or Lambda.
