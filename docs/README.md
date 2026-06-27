# Cache Warmup Recommender — Documentação Técnica

**Projeto:** Intelligent Fallback Cache — AWS  
**Versão:** 1.0  
**Data:** 2026-05  
**Stack:** AWS SageMaker Processing · S3 · Lambda · Python 3.10

---

## Índice

1. [Visão Geral da Solução](#1-visão-geral-da-solução)
2. [Arquitetura](#2-arquitetura)
3. [Estrutura dos Dados de Entrada](#3-estrutura-dos-dados-de-entrada)
4. [Metodologia e Fórmula Matemática](#4-metodologia-e-fórmula-matemática)
5. [Dimensões de Recomendação](#5-dimensões-de-recomendação)
6. [Estrutura do JSON de Saída](#6-estrutura-do-json-de-saída)
7. [Implementação no SageMaker — Passo a Passo](#7-implementação-no-sagemaker--passo-a-passo)
8. [Configuração das Lambdas](#8-configuração-das-lambdas)
9. [Parâmetros e Tunagem do Modelo](#9-parâmetros-e-tunagem-do-modelo)
10. [Monitoramento e Operação](#10-monitoramento-e-operação)
11. [Resultados de Exemplo](#11-resultados-de-exemplo)
12. [Estrutura de Arquivos do Repositório](#12-estrutura-de-arquivos-do-repositório)

---

## 1. Visão Geral da Solução

O **Cache Warmup Recommender** é um modelo de recomendação que analisa logs de consulta à API Polaris/Experian e determina quais combinações de **cliente → documento → relatório** têm maior probabilidade de serem consultadas novamente em breve. Essas recomendações são usadas para pré-aquecer um **Cache de Fallback ativo** na AWS, garantindo disponibilidade mesmo quando as APIs de origem estão offline.

### Problema resolvido

Sem um warmup inteligente, o cache só seria útil após a primeira consulta real — exatamente quando o usuário já sofreu a latência (ou o erro de indisponibilidade). Com o modelo, o cache é populado **antes** das consultas chegarem, com base em padrões históricos.

### Fluxo resumido

```
CSV de logs → S3 → Lambda Trigger → SageMaker Processing Job
  → JSON de recomendações → S3 → Lambda Warmup → APIs → Cache
```

---

## 2. Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS Cloud                                 │
│                                                                  │
│  ┌──────────────┐    S3 Event     ┌─────────────────────────┐   │
│  │  S3 (input)  │ ─────────────► │  Lambda 1 — Trigger     │   │
│  │  data/*.csv  │                 │  (lambda_trigger.py)    │   │
│  └──────────────┘                 └──────────┬──────────────┘   │
│                                              │ CreateProcessingJob│
│                                              ▼                   │
│                                  ┌───────────────────────────┐   │
│                                  │  SageMaker Processing Job  │   │
│                                  │  (recommender.py)          │   │
│                                  │  ml.m5.xlarge              │   │
│                                  │  sklearn:1.2 image         │   │
│                                  └──────────┬────────────────┘   │
│                                             │ PutObject           │
│                                             ▼                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  S3 (output)                                               │  │
│  │  output/warmup/warmup_recommendations.json                 │  │
│  └───────────────────────────┬────────────────────────────────┘  │
│                              │ S3 Event (ObjectCreated)           │
│                              ▼                                    │
│                  ┌───────────────────────┐                       │
│                  │  Lambda 2 — Warmup    │                       │
│                  │  (lambda_warmup.py)   │                       │
│                  └──────────┬────────────┘                       │
│                             │ HTTP calls                          │
│                             ▼                                     │
│                  ┌───────────────────────┐                       │
│                  │  APIs Experian/Polaris │                       │
│                  └──────────┬────────────┘                       │
│                             │                                     │
│                             ▼                                     │
│                  ┌───────────────────────┐                       │
│                  │  Cache (ElastiCache,  │                       │
│                  │  API Gateway Cache,   │                       │
│                  │  ou CloudFront)       │                       │
│                  └───────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

### Componentes

| Componente | Serviço AWS | Função |
|---|---|---|
| Armazenamento de logs | S3 (bucket input) | Recebe os CSVs de log exportados |
| Trigger | Lambda + S3 Event | Inicia o job ao detectar novo CSV |
| Processamento ML | SageMaker Processing Job | Executa o modelo de recomendação |
| Armazenamento de recomendações | S3 (bucket output) | Persiste o JSON de warmup |
| Warmup | Lambda + S3 Event | Lê o JSON e chama as APIs |
| Cache | ElastiCache / API GW | Armazena as respostas pré-aquecidas |

---

## 3. Estrutura dos Dados de Entrada

O modelo consome um arquivo CSV com separador `|` contendo os seguintes campos:

| Campo | Tipo | Descrição |
|---|---|---|
| `ID_REPORT` | int64 | Identificador único do relatório |
| `reportName` | string | Nome do relatório (ex: `RELATORIO_AVANCADO_PJ`) |
| `TYPE_REPORT` | string | Tipo: `PJ` (pessoa jurídica) ou `PF` (pessoa física) |
| `ID_FEATURE` | int64 | Identificador da feature |
| `FEATURENAME` | string | Nome da feature (ex: `PARTICIPACOES`) |
| `FEATURE_TYPE` | string | Tipo: `O` (output) ou `R` (reference) |
| `channel` | string | Canal de acesso (ex: `INTERNAL`) |
| `billing` | bool | Indica se a consulta foi cobrada |
| `inquiry` | bool | Indica se foi uma consulta de crédito |
| `post_report_view` | bool | Indica se o relatório foi visualizado |
| `httpStatus` | int | Código HTTP da resposta da API |
| `httpTime` | int | Latência da chamada em milissegundos |
| `inclusionDate` | datetime | Timestamp da consulta |
| `customerDocument` | int64 | CNPJ/CPF do cliente que fez a consulta |
| `consultedDocument` | string (hash) | Hash do CNPJ/CPF consultado |

### Exemplo de registro

```
ID_REPORT          = 6631097597580458492
reportName         = RELATORIO_AVANCADO_PJ_PME
TYPE_REPORT        = PJ
ID_FEATURE         = 3065482819
FEATURENAME        = DIVIDAS_ORGAOS_PUBLICOS
FEATURE_TYPE       = O
billing            = True
inquiry            = True
post_report_view   = True
httpStatus         = 200
httpTime           = 2475
inclusionDate      = 2026-05-14 14:32:41
customerDocument   = 66251314004156
consultedDocument  = 4e412d7cd1dc4e579e9a0d25a15035c087ccba6f
```

---

## 4. Metodologia e Fórmula Matemática

### 4.1 Fórmula central — Média Ponderada

```
score(e) = α · V_norm(e)  +  β · R(e, t)  +  γ · W_biz(e)
```

onde **α + β + γ = 1.0** (pesos configúráveis).

### 4.2 Componente de Volume — V_norm

```
V_norm(e) = volume(e) / max_volume
```

- `volume(e)`: número de vezes que o item `e` foi consultado na janela temporal
- `max_volume`: maior volume observado em qualquer item da mesma dimensão
- Resultado em **[0, 1]** — normalização min-max com mínimo = 0

**Interpretação:** itens muito consultados recebem score de volume próximo a 1. Um relatório com 100.000 acessos tem V_norm ≈ 1.0; um com 1.000 acessos tem V_norm ≈ 0.01.

### 4.3 Componente de Recência — R (Decaimento Exponencial)

```
R(e, t) = exp(-λ · Δt_dias)
```

- `Δt_dias`: dias entre a última consulta ao item e a data de referência (máximo do dataset)
- `λ` (lambda): taxa de decaimento — controla a velocidade com que itens antigos perdem relevância

**Tabela de decaimento (λ = 0.15):**

| Δt | R(e, t) | Interpretação |
|---|---|---|
| 0 dias (hoje) | 1.000 | Máxima relevância |
| 1 dia | 0.861 | Ligeira queda |
| 3 dias | 0.638 | Moderadamente recente |
| 7 dias | 0.350 | Meia-vida aproximada |
| 14 dias | 0.122 | Item "frio" |
| 30 dias | 0.011 | Irrelevante |

**Por que exponencial?** O acesso a um documento de crédito corporativo tem padrão de "rajada" — a mesma empresa é consultada intensamente em um período (análise de crédito em andamento) e depois cessa. O decaimento exponencial captura exatamente esse comportamento: quanto mais recente o acesso, maior a probabilidade de repetição.

### 4.4 Componente de Peso de Negócio — W_biz

```
W_biz(e) = 0.30 · billing_rate
         + 0.25 · inquiry_rate
         + 0.25 · post_view_rate
         + 0.20 · latency_norm
```

| Sub-componente | Fórmula | Significado |
|---|---|---|
| `billing_rate` | média de `billing == True` | Proporção de consultas que geraram cobrança — alta lucratividade |
| `inquiry_rate` | média de `inquiry == True` | Proporção de consultas de crédito — alta criticidade |
| `post_view_rate` | média de `post_report_view == True` | Proporção de relatórios efetivamente visualizados — alta utilidade |
| `latency_norm` | `httpTime_médio / httpTime_máx` | Chamadas lentas têm **mais** a ganhar com o cache |

**Por que incluir latência?** Uma consulta que demora 5s em média tem impacto de UX muito maior do que uma que demora 300ms. Pré-aquecê-la no cache elimina esse gargalo, gerando mais valor para o negócio.

### 4.5 Pesos padrão e racional

| Parâmetro | Valor padrão | Racional |
|---|---|---|
| α (volume) | **0.35** | Volume é um sinal forte mas pode ser inflado por um único cliente atípico |
| β (recência) | **0.40** | Maior peso — o cache deve refletir o padrão de uso atual, não histórico |
| γ (negócio) | **0.25** | Sinal qualitativo importante mas subordinado ao padrão de acesso real |
| λ (decaimento) | **0.15** | Meia-vida ≈ 4.6 dias — compatível com ciclos de análise de crédito B2B |

---

## 5. Dimensões de Recomendação

O modelo gera recomendações em 5 dimensões independentes:

### 5.1 `top_reports` — Relatórios para warmup global
Responde: *"Quais relatórios devo manter sempre quentes no cache?"*

Agrupamento por `(reportName, TYPE_REPORT)`. Ideal para configurar regras de TTL por tipo de relatório.

### 5.2 `top_features` — Features mais demandadas
Responde: *"Quais features específicas são mais consultadas?"*

Agrupamento por `(ID_FEATURE, FEATURENAME, FEATURE_TYPE)`. Útil para otimizar o payload das respostas cacheadas.

### 5.3 `top_customers` — Clientes prioritários
Responde: *"Quais CNPJs/CPFs consultantes têm maior demanda esperada?"*

Agrupamento por `customerDocument`. Permite warmup proativo por cliente de alto valor.

### 5.4 `top_consulted_documents` — Documentos mais consultados
Responde: *"Quais CNPJs/CPFs são mais consultados como alvo?"*

Agrupamento por `consultedDocument` (hash). Identifica empresas-alvo que são objeto frequente de análise.

### 5.5 `top_pairs` — Pares customer → document (warmup direcionado)
Responde: *"Dado que o cliente X vai acessar o sistema, qual documento ele provavelmente vai consultar?"*

Agrupamento por `(customerDocument, consultedDocument)`. Esta é a recomendação mais precisa e diretamente acionável para a Lambda de warmup.

---

## 6. Estrutura do JSON de Saída

Arquivo: `s3://<OUTPUT_BUCKET>/<OUTPUT_PREFIX>/warmup_recommendations.json`

```json
{
  "generated_at": "2026-05-14T23:59:07.158728",
  "model_params": {
    "formula": "score = α·V_norm + β·exp(-λ·Δt) + γ·W_biz",
    "W_biz_formula": "W = 0.30·billing + 0.25·inquiry + 0.25·post_view + 0.20·latency_norm",
    "alpha": 0.35,
    "beta": 0.40,
    "gamma": 0.25,
    "lambda": 0.15
  },
  "warmup_targets": {
    "top_reports": [
      {
        "reportName": "RELATORIO_AVANCADO_TOP_SCORE_PJ",
        "TYPE_REPORT": "PJ",
        "volume": 102007,
        "score": 0.7647
      }
    ],
    "top_features": [
      {
        "ID_FEATURE": "3005475493",
        "FEATURENAME": "PARTICIPACOES",
        "FEATURE_TYPE": "O",
        "volume": 47085,
        "score": 0.7123
      }
    ],
    "top_customers": [
      {
        "customerDocument": "93986088673863",
        "volume": 111622,
        "score": 0.7039
      }
    ],
    "top_consulted_documents": [
      {
        "consultedDocument": "b4b3f7ab82d4bb59b36cf8cd4d11679e32d34901",
        "volume": 56327,
        "score": 0.6977
      }
    ],
    "top_pairs": [
      {
        "customerDocument": "93986088673863",
        "consultedDocument": "b4b3f7ab82d4bb59b36cf8cd4d11679e32d34901",
        "volume": 56327,
        "score": 0.6965
      }
    ]
  },
  "stats": {
    "total_records": 724753,
    "unique_customers": 21412,
    "unique_consulted": 276314,
    "unique_reports": 18,
    "unique_features": 66,
    "date_range": { "from": "2026-05-01", "to": "2026-05-14" }
  }
}
```

---

## 7. Implementação no SageMaker — Passo a Passo

### Pré-requisitos

- Conta AWS com acesso a SageMaker, S3, Lambda, IAM
- AWS CLI configurado (`aws configure`)
- Python 3.10+ e `boto3` instalados localmente

---

### Passo 1 — Criar os Buckets S3

```bash
# Bucket principal (input + output + code)
aws s3 mb s3://meu-bucket-polaris --region us-east-1

# Criar as "pastas" (prefixos)
aws s3api put-object --bucket meu-bucket-polaris --key input/
aws s3api put-object --bucket meu-bucket-polaris --key output/warmup/
aws s3api put-object --bucket meu-bucket-polaris --key code/
```

**Estrutura de pastas:**

```
meu-bucket-polaris/
├── input/           ← CSVs de log depositados aqui
├── output/warmup/   ← JSON de recomendações gerado aqui
└── code/            ← Script Python do modelo
```

---

### Passo 2 — Criar a IAM Role para o SageMaker

No console AWS → IAM → Roles → Create Role:

1. **Trusted entity:** `sagemaker.amazonaws.com`
2. **Nome da role:** `SageMakerExecutionRole`
3. **Políticas gerenciadas a anexar:**
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess` (ou a política customizada do arquivo `iam_policy.json`)

```bash
# Via CLI — criar a trust policy
cat > trust-policy.json << 'JSON'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "sagemaker.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
JSON

aws iam create-role \
  --role-name SageMakerExecutionRole \
  --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

---

### Passo 3 — Fazer upload do script para o S3

```bash
aws s3 cp recommender.py s3://meu-bucket-polaris/code/recommender.py
```

---

### Passo 4 — Testar o Processing Job manualmente (primeira vez)

```bash
# Upload do CSV de teste
aws s3 cp seu_arquivo.csv s3://meu-bucket-polaris/input/data.csv

# Executar o job
python run_job.py
```

Acompanhe pelo console: **SageMaker → Processing Jobs** ou via CLI:

```bash
aws sagemaker describe-processing-job \
  --processing-job-name cache-warmup-20260514-120000 \
  --query "ProcessingJobStatus"
```

Após a conclusão (`Completed`), o JSON estará em:
```
s3://meu-bucket-polaris/output/warmup/warmup_recommendations.json
```

Verifique:
```bash
aws s3 cp s3://meu-bucket-polaris/output/warmup/warmup_recommendations.json - | python3 -m json.tool | head -50
```

---

### Passo 5 — Configurar Lambda Trigger (S3 → SageMaker)

#### 5a. Criar a Lambda

No console AWS → Lambda → Create Function:

- **Nome:** `cache-warmup-trigger`
- **Runtime:** Python 3.12
- **Role:** crie uma nova com permissão `AmazonSageMakerFullAccess` + `AmazonS3ReadOnlyAccess` + `IAMPassRole`

Faça upload do arquivo `lambda_trigger.py` como código da função.

**Variáveis de ambiente da Lambda:**

| Variável | Valor |
|---|---|
| `SAGEMAKER_ROLE_ARN` | `arn:aws:iam::123456789012:role/SageMakerExecutionRole` |
| `OUTPUT_BUCKET` | `meu-bucket-polaris` |
| `OUTPUT_PREFIX` | `output/warmup/` |
| `CODE_S3_URI` | `s3://meu-bucket-polaris/code/` |

#### 5b. Adicionar o trigger S3

No console da Lambda → Add Trigger:

- **Source:** S3
- **Bucket:** `meu-bucket-polaris`
- **Event type:** `s3:ObjectCreated:*`
- **Prefix:** `input/`
- **Suffix:** `.csv`

---

### Passo 6 — Configurar Lambda Warmup (S3 → APIs)

#### 6a. Criar a Lambda

- **Nome:** `cache-warmup-executor`
- **Runtime:** Python 3.12
- **Timeout:** 5 minutos (o default de 3s não é suficiente)
- **Memory:** 256 MB

Faça upload do arquivo `lambda_warmup.py`.

**Variáveis de ambiente:**

| Variável | Valor |
|---|---|
| `API_BASE_URL` | `https://api.experian.com.br/v1` |
| `API_KEY` | `<sua-api-key>` (use Secrets Manager em produção) |
| `MAX_PAIRS` | `500` |
| `TIMEOUT_SECONDS` | `10` |

#### 6b. Adicionar o trigger S3

- **Bucket:** `meu-bucket-polaris`
- **Event type:** `s3:ObjectCreated:*`
- **Prefix:** `output/warmup/`
- **Suffix:** `.json`

---

### Passo 7 — Agendar execução periódica (opcional)

Para rodas o modelo diariamente mesmo sem novo CSV:

No console → **EventBridge → Rules → Create Rule:**

```json
{
  "source": ["aws.events"],
  "schedule": "cron(0 6 * * ? *)"
}
```

Target: Lambda `cache-warmup-trigger` com input customizado:
```json
{
  "Records": [{
    "s3": {
      "bucket": {"name": "meu-bucket-polaris"},
      "object": {"key": "input/data.csv"}
    }
  }]
}
```

---

## 8. Configuração das Lambdas

### Lambda 1 — Trigger (`lambda_trigger.py`)

Responsabilidade: detectar novo CSV no S3 e iniciar o SageMaker Processing Job.

```
Evento S3 (ObjectCreated)
    └─► Extrai bucket + key do evento
    └─► Valida extensão .csv
    └─► Chama sm.create_processing_job(...)
    └─► Retorna nome do job
```

### Lambda 2 — Warmup (`lambda_warmup.py`)

Responsabilidade: ler o JSON de recomendações e chamar as APIs para popular o cache.

```
Evento S3 (ObjectCreated: warmup_recommendations.json)
    └─► Lê o JSON do S3
    └─► Itera top_pairs (limitado por MAX_PAIRS)
    └─► Para cada par: GET /reports/{reportName}?customer=X&consulted=Y
    └─► Contabiliza ok/erro e loga no CloudWatch
```

**Recomendações para produção:**

- Use **AWS Secrets Manager** para a `API_KEY` em vez de variável de ambiente
- Adicione **retry com backoff exponencial** nas chamadas de API
- Use **SQS** como buffer entre o JSON de recomendações e as chamadas de API, para não ultrapassar o timeout da Lambda em volumes altos
- Implemente **dead-letter queue (DLQ)** para capturar falhas

---

## 9. Parâmetros e Tunagem do Modelo

Todos os parâmetros são injetados como variáveis de ambiente no Processing Job.

### Ajustando os pesos (α, β, γ)

| Cenário | α | β | γ | Efeito |
|---|---|---|---|---|
| **Padrão** | 0.35 | 0.40 | 0.25 | Balanceado |
| **Priorizar frequência absoluta** | 0.55 | 0.25 | 0.20 | Itens muito acessados dominam |
| **Priorizar atualidade** | 0.20 | 0.60 | 0.20 | Apenas consultas recentes importam |
| **Priorizar valor de negócio** | 0.20 | 0.30 | 0.50 | Foco em billing e consultas críticas |

### Ajustando o decaimento (λ)

| λ | Meia-vida | Use quando |
|---|---|---|
| 0.05 | ~14 dias | Padrões estáveis, contratos longos |
| **0.15** | **~4.6 dias** | **Padrão — análises de crédito B2B** |
| 0.30 | ~2.3 dias | Alta volatilidade, queries de monitoramento diário |
| 0.70 | ~1 dia | Apenas o último dia importa |

### Ajustando o volume de recomendações

| Parâmetro | Default | Aumentar se |
|---|---|---|
| `TOP_N_REPORTS` | 20 | Há mais de 20 relatórios distintos em uso ativo |
| `TOP_N_PAIRS` | 2000 | Cache tem capacidade para mais entradas |
| `TOP_N_PAIRS_CUST` | 200 | Há muitos clientes de médio volume (cauda longa) |

---

## 10. Monitoramento e Operação

### CloudWatch — Métricas a acompanhar

| Métrica | Como verificar |
|---|---|
| Duração do Processing Job | SageMaker → Processing Jobs → Duration |
| Erros da Lambda de warmup | CloudWatch → Log Groups → `/aws/lambda/cache-warmup-executor` |
| Taxa de cache hit | Depende da implementação do cache (ElastiCache metrics) |
| Volume de CSVs processados | S3 → Metrics → NumberOfObjects |

### Alarme recomendado (CloudWatch Alarm)

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name "cache-warmup-job-failed" \
  --metric-name "FailedJobCount" \
  --namespace "AWS/SageMaker" \
  --statistic Sum \
  --period 3600 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --alarm-actions arn:aws:sns:us-east-1:123456789012:alertas
```

### Checklist de operação diária

- [ ] Processing Job concluiu com status `Completed`
- [ ] JSON de saída foi atualizado no S3
- [ ] Lambda de warmup executou sem erros (CloudWatch)
- [ ] Taxa de cache hit aumentou após o warmup
- [ ] Nenhum custo anômalo no SageMaker (Processing Jobs)

---

## 11. Resultados de Exemplo

### Top relatórios (dataset 2026-05-01 a 2026-05-14, 724.753 registros)

| # | Relatório | Tipo | Volume | Score |
|---|---|---|---|---|
| 1 | RELATORIO_AVANCADO_TOP_SCORE_PJ | PJ | 102.007 | 0.7647 |
| 2 | RELATORIO_AVANCADO_PJ | PJ | 72.632 | 0.6528 |
| 3 | RELATORIO_INTERMEDIARIO_TOP_SCORE_PJ | PJ | 68.581 | 0.6515 |
| 4 | RELATORIO_INTERMEDIARIO_PJ | PJ | 71.219 | 0.6423 |
| 5 | RELATORIO_AVANCADO_PF_NOVO_FRONT | PF | 63.154 | 0.6268 |
| 6 | RELATORIO_AVANCADO_PJ_ANALITICO | PJ | 58.612 | 0.6045 |
| 7 | RELATORIO_AVANCADO_PJ_PME | PJ | 55.429 | 0.5984 |

> Os 7 primeiros relatórios respondem por **~72%** do volume total. Concentrar o warmup neles oferece a maior cobertura de cache com menor número de chamadas de API.

### Top par (warmup mais prioritário)

```
customerDocument  : 93986088673863
consultedDocument : b4b3f7ab82d4bb59b36cf8cd4d11679e32d34901
Relatório         : RELATORIO_INTERMEDIARIO_PJ
Volume            : 32.145 consultas em 14 dias (~2.300/dia)
Score             : 0.6965
```

Este par sozinho representa **4,4%** de todo o volume do dataset — um único cliente consultando o mesmo documento repetidamente. É o candidato mais evidente para cache dedicado.

---

## 12. Estrutura de Arquivos do Repositório

```
cache-warmup-recommender/
├── recommender.py          ← Script principal (SageMaker Processing Job)
├── run_job.py              ← Aciona o job manualmente ou via Lambda
├── lambda_trigger.py       ← Lambda 1: S3 Event → SageMaker Job
├── lambda_warmup.py        ← Lambda 2: JSON → chamadas de API → Cache
├── iam_policy.json         ← Política IAM para a role do SageMaker
├── iam_lambda_policy.json  ← Política IAM para as Lambdas
└── DOCUMENTACAO_CACHE_WARMUP_RECOMMENDER.md  ← Este arquivo
```

---

*Documentação gerada em 2026-05 · Cache Warmup Recommender v1.0*
