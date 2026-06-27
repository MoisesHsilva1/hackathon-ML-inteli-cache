# Project Structure

```
cache-warmup-recommender/
├── src/
│   ├── recommender.py          # Core scoring model (SageMaker Processing Job)
│   └── recommender_ml.py       # ML-enhanced hybrid model (SVD + heuristic)
├── lambdas/
│   ├── trigger.py              # S3 CSV upload → starts SageMaker job
│   ├── warmup.py               # Reads recommendation JSON → calls APIs
│   └── run_job.py              # Manual launcher / Lambda handler
├── iam/
│   ├── sagemaker_policy.json   # IAM policy for SageMaker execution role
│   └── lambda_policy.json      # IAM policy for Lambda functions
├── data/
│   ├── sample_input.csv        # Anonymized input data
│   └── sample_output.json      # Sample recommendation output
├── docs/
│   └── README.md               # Full technical documentation (Portuguese)
├── .github/
│   └── workflows/
│       └── main.yml
└── .kiro/
```

## Architecture Layers

1. **Data Ingestion** — CSV logs land in S3 `input/` prefix
2. **Orchestration** — `lambdas/trigger.py` detects new CSV and starts the SageMaker job
3. **ML Processing** — `src/recommender.py` scores entities across 5 dimensions
4. **Output** — JSON recommendations written to S3 `output/warmup/` prefix
5. **Execution** — `lambdas/warmup.py` reads recommendations and calls external APIs

## S3 Bucket Layout

```
<bucket>/
├── input/           # Raw CSV log files
├── output/warmup/   # Generated recommendation JSON
└── code/            # recommender.py deployed for SageMaker
```

## Conventions

- Each Lambda is a single-file module with a `lambda_handler(event, context)` entry point
- `run_job.py` serves dual purpose: CLI script (`__main__`) and Lambda handler
- Configuration via environment variables — no config files
- IAM policies live in `iam/` folder as separate JSON files
