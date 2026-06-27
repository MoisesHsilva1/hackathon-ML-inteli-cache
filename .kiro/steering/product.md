# Cache Warmup Recommender

An intelligent cache pre-warming system for the Experian/Polaris credit report API. Analyzes historical query logs to predict which customer-document-report combinations are most likely to be requested again soon, then proactively populates a fallback cache before actual requests arrive.

## Problem Solved

Without intelligent warmup, a fallback cache only helps after the first real query — the exact moment a user already experienced latency or an availability error. This system pre-populates the cache based on historical patterns so responses are ready before users ask.

## Core Flow

```
CSV logs → S3 → Lambda Trigger → SageMaker Processing Job → JSON recommendations → S3 → Lambda Warmup → APIs → Cache
```

## Domain Context

- Operates in the Brazilian credit bureau (bureau de crédito) domain
- Handles PJ (pessoa jurídica / corporate) and PF (pessoa física / individual) report types
- Sensitive data: customer documents (CNPJ/CPF) are present in logs and recommendations
- Documentation and code comments are written in Brazilian Portuguese
