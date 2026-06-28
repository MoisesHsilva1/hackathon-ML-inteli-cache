# Cache Warmup Recommender — Documentação Técnica Completa

**Projeto:** Intelligent Fallback Cache — AWS
**Versão:** 2.0 (Hybrid SVD)
**Data:** 2026-05
**Stack:** AWS SageMaker Processing · S3 · Lambda · Python 3.10

---

## Índice

1. [Visão Geral da Solução](#1-visão-geral-da-solução)
2. [Arquitetura](#2-arquitetura)
3. [Estrutura dos Dados de Entrada](#3-estrutura-dos-dados-de-entrada)
4. [Modelo de Recomendação — Abordagem Híbrida](#4-modelo-de-recomendação--abordagem-híbrida)
5. [Metodologia e Fórmula Matemática](#5-metodologia-e-fórmula-matemática)
6. [Dimensões de Recomendação](#6-dimensões-de-recomendação)
7. [Pipeline de Processamento](#7-pipeline-de-processamento)
8. [Estrutura do JSON de Saída](#8-estrutura-do-json-de-saída)
9. [Métricas do Modelo](#9-métricas-do-modelo)
10. [Parâmetros e Tunagem do Modelo](#10-parâmetros-e-tunagem-do-modelo)
11. [Implementação no SageMaker — Passo a Passo](#11-implementação-no-sagemaker--passo-a-passo)
12. [Configuração das Lambdas](#12-configuração-das-lambdas)
13. [Infraestrutura AWS](#13-infraestrutura-aws)
14. [Monitoramento e Operação](#14-monitoramento-e-operação)
15. [Custos Estimados](#15-custos-estimados)
16. [Decisões de Design](#16-decisões-de-design)
17. [Glossário de Termos](#17-glossário-de-termos)
18. [Resultados de Exemplo](#18-resultados-de-exemplo)
19. [Comandos Úteis](#19-comandos-úteis)
20. [Estrutura de Arquivos do Repositório](#20-estrutura-de-arquivos-do-repositório)

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
│  │  data/*.csv  │                 │  (trigger.py)           │   │
│  └──────────────┘                 └──────────┬──────────────┘   │
│                                              │ CreateProcessingJob│
│                                              ▼                   │
│                                  ┌───────────────────────────┐   │
│                                  │  SageMaker Processing Job  │   │
│                                  │  (recommender_ml.py)       │   │
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
│                  │  (warmup.py)          │                       │
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

### Fluxo Event-Driven

1. **Ingestão**: CSV com logs de acesso é depositado em `s3://bucket/input/`
2. **Trigger**: Lambda detecta o evento S3 e inicia um SageMaker Processing Job
3. **Processamento ML**: O job executa o modelo híbrido (SVD + Heurístico)
4. **Output**: JSON com recomendações é gravado em `s3://bucket/output/warmup/`
5. **Warmup**: Segunda Lambda detecta o JSON e chama as APIs para pré-aquecer o cache

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

## 4. Modelo de Recomendação — Abordagem Híbrida

### Abordagem Híbrida (v2.0)

O modelo combina duas técnicas complementares:

| Componente | Técnica | Propósito |
|-----------|---------|-----------|
| **Matrix Factorization** | SVD (Truncated) | Captura padrões latentes de co-ocorrência |
| **Score de Negócio** | Heurística ponderada | Incorpora volume, recência e sinais de negócio |

### Fórmula do Score Final

```
score_final = W_MF · score_mf + W_BIZ · score_biz
```

Onde:
- `W_MF = 0.45` — peso do collaborative filtering
- `W_BIZ = 0.55` — peso do score de negócio

---

## 5. Metodologia e Fórmula Matemática

### 5.1 Fórmula central — Média Ponderada (Score de Negócio)

```
biz_score(e) = α · V_norm(e)  +  β · R(e, t)  +  γ · W_biz(e)
```

onde **α + β + γ = 1.0** (pesos configuráveis).

### 5.2 Componente de Volume — V_norm

```
V_norm(e) = volume(e) / max_volume
```

- `volume(e)`: número de vezes que o item `e` foi consultado na janela temporal
- `max_volume`: maior volume observado em qualquer item da mesma dimensão
- Resultado em **[0, 1]** — normalização min-max com mínimo = 0

**Interpretação:** itens muito consultados recebem score de volume próximo a 1. Um relatório com 100.000 acessos tem V_norm ≈ 1.0; um com 1.000 acessos tem V_norm ≈ 0.01.

### 5.3 Componente de Recência — R (Decaimento Exponencial)

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

### 5.4 Componente de Peso de Negócio — W_biz

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

### 5.5 Pesos padrão e racional

| Parâmetro | Valor padrão | Racional |
|---|---|---|
| α (volume) | **0.35** | Volume é um sinal forte mas pode ser inflado por um único cliente atípico |
| β (recência) | **0.40** | Maior peso — o cache deve refletir o padrão de uso atual, não histórico |
| γ (negócio) | **0.25** | Sinal qualitativo importante mas subordinado ao padrão de acesso real |
| λ (decaimento) | **0.15** | Meia-vida ≈ 4.6 dias — compatível com ciclos de análise de crédito B2B |

---

## 6. Dimensões de Recomendação

O modelo gera recomendações em 5 dimensões independentes:

### 6.1 `top_reports` — Relatórios para warmup global
Responde: *"Quais relatórios devo manter sempre quentes no cache?"*

Agrupamento por `(reportName, TYPE_REPORT)`. Ideal para configurar regras de TTL por tipo de relatório.

### 6.2 `top_features` — Features mais demandadas
Responde: *"Quais features específicas são mais consultadas?"*

Agrupamento por `(ID_FEATURE, FEATURENAME, FEATURE_TYPE)`. Útil para otimizar o payload das respostas cacheadas.

### 6.3 `top_customers` — Clientes prioritários
Responde: *"Quais CNPJs/CPFs consultantes têm maior demanda esperada?"*

Agrupamento por `customerDocument`. Permite warmup proativo por cliente de alto valor.

### 6.4 `top_consulted_documents` — Documentos mais consultados
Responde: *"Quais CNPJs/CPFs são mais consultados como alvo?"*

Agrupamento por `consultedDocument` (hash). Identifica empresas-alvo que são objeto frequente de análise.

### 6.5 `top_pairs` — Pares customer → document (warmup direcionado)
Responde: *"Dado que o cliente X vai acessar o sistema, qual documento ele provavelmente vai consultar?"*

Agrupamento por `(customerDocument, consultedDocument)`. Esta é a recomendação mais precisa e diretamente acionável para a Lambda de warmup.

---

## 7. Pipeline de Processamento

### Etapa 1: Carga de Dados (`load_data`)

- Lê todos os CSVs do diretório de input (separador `|`)
- Concatena em um único DataFrame
- Valida presença de pelo menos um arquivo

### Etapa 2: Preparação (`prepare`)

- Converte tipos (datas, booleanos, numéricos)
- Limpa strings (remove espaços, aspas)
- Calcula `delta_days` — dias desde a consulta até a data mais recente
- Define `t_ref` como data de referência (max inclusionDate)

### Etapa 3: Matriz de Interações (`build_interaction_matrix`)

- Agrupa por (customerDocument, consultedDocument)
- Conta interações por par
- Aplica `log1p` para suavizar distribuição power-law
- Constrói matriz CSR esparsa

### Etapa 4: Treinamento SVD (`train_svd`)

- Executa TruncatedSVD na matriz esparsa
- Gera `user_factors` (representação dos customers)
- Gera `item_factors` (representação dos documents)
- Registra variância explicada

### Etapa 5: Avaliação (`evaluate_model`)

- Amostra até 5000 users
- Reconstrói a matriz via `user_factors @ item_factors.T`
- Calcula RMSE apenas nas posições não-zero (evita viés de zeros)

### Etapa 6: Score Híbrido (`compute_hybrid_pairs`)

- Calcula score de negócio para cada par
- Calcula score MF (dot product dos fatores latentes)
- Normaliza ambos para [0, 1]
- Combina: `score = 0.45 · mf_score + 0.55 · biz_score`

### Etapa 7: Geração de Recomendações (`generate_recommendations`)

- Gera top-N para 5 dimensões: reports, features, customers, documents, pairs
- Monta payload JSON completo com métricas e metadados
- Grava em `/opt/ml/processing/output/warmup_recommendations.json`

---

## 8. Estrutura do JSON de Saída

Arquivo: `s3://<OUTPUT_BUCKET>/<OUTPUT_PREFIX>/warmup_recommendations.json`

```json
{
  "generated_at": "2026-05-14T23:59:07.158728+00:00",
  "model_version": "2.0-hybrid-svd",
  "model_params": {
    "formula": "score_final = w_mf · score_mf + w_biz · (α·V_norm + β·exp(-λ·Δt) + γ·W_biz)",
    "alpha": 0.35,
    "beta": 0.40,
    "gamma": 0.25,
    "lambda": 0.15,
    "w_mf": 0.45,
    "w_biz": 0.55,
    "svd_components": 50,
    "svd_iterations": 15
  },
  "model_metrics": {
    "svd_explained_variance_pct": 85.42,
    "svd_reconstruction_rmse": 0.1234,
    "matrix_shape": [5000, 3000],
    "matrix_density_pct": 0.85,
    "n_interactions": 127500,
    "training_time_seconds": 3.5
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

## 9. Métricas do Modelo

### Métricas de Qualidade do SVD

| Métrica | O que mede | Valor bom | Como interpretar |
|---------|-----------|-----------|------------------|
| **Variância Explicada (%)** | Quanto da informação original o SVD capturou | > 60% | Análogo ao R² em regressão. 80% = o modelo captura 80% dos padrões |
| **RMSE de Reconstrução** | Erro médio ao reconstruir a matriz original | < 0.3 | Quanto menor, melhor o modelo reproduz os dados observados |
| **Densidade da Matriz (%)** | Proporção de células não-zero na matriz | 0.01% - 5% | Matrizes muito esparsas (< 0.01%) dificultam o aprendizado |
| **N Interações** | Total de pares (customer, document) observados | — | Mais interações = mais dados para o modelo aprender |
| **Tempo de Treinamento (s)** | Duração do fit do SVD | < 60s | Indica se o modelo é viável para re-treino frequente |

### Métricas de Impacto (Warmup)

| Métrica | Definição | Cálculo |
|---------|-----------|---------|
| **Cache Hit Rate** | % de requisições servidas pelo cache após warmup | `requests_from_cache / total_requests × 100` |
| **Cobertura** | % dos customers ativos cobertos pelas recomendações | `pairs_recomendados / customers_ativos × 100` |
| **Latência P95** | Tempo de resposta no percentil 95 | Com cache: ~50ms, Sem cache: ~800ms |
| **Redução de Chamadas** | Economia de chamadas ao backend real | `1 - (calls_backend_com_cache / calls_backend_sem_cache)` |

### Interpretação das Métricas

**Variância Explicada:**
- **O que é**: Percentual da informação original que o SVD conseguiu capturar com N componentes
- **Analogia**: Como compressão de imagem — 85% significa que você mantém 85% da qualidade com muito menos dados
- **Bom**: > 60%. Acima de 80% é excelente
- **Ruim**: < 40%. O modelo não está capturando padrões suficientes
- **Ação se ruim**: Aumentar `N_COMPONENTS` ou verificar se os dados têm padrões reais

**RMSE de Reconstrução:**
- **O que é**: Erro quadrático médio ao tentar reconstruir a matriz original a partir dos fatores latentes
- **Analogia**: Quanto "borrada" fica a imagem após compressão e descompressão
- **Bom**: < 0.3
- **Aceitável**: 0.3 - 0.6
- **Ruim**: > 0.6
- **Ação se ruim**: Mais componentes, mais iterações, ou dados insuficientes

**Densidade da Matriz:**
- **O que é**: Proporção de células preenchidas na matriz customers × documents
- **Contexto**: Em sistemas de recomendação, matrizes são tipicamente 99%+ vazias
- **Muito esparsa** (< 0.01%): Pode haver poucos dados para aprender padrões confiáveis
- **Normal** (0.01% - 5%): Faixa típica para collaborative filtering funcionar bem
- **Densa** (> 5%): Incomum mas positivo — mais sinais disponíveis

**Score Híbrido:**
- **Range**: 0.0 a ~1.0 (normalizado)
- **> 0.7**: Altamente provável de ser consultado novamente (prioridade máxima para warmup)
- **0.4 - 0.7**: Probabilidade moderada (incluir se houver capacidade)
- **< 0.4**: Baixa probabilidade (pode ser ignorado no warmup)

---

## 10. Parâmetros e Tunagem do Modelo

Todos os parâmetros são injetados como variáveis de ambiente no Processing Job.

### Pesos do Score de Negócio

| Parâmetro | Valor | Env Var | Descrição |
|-----------|-------|---------|-----------|
| **α (Alpha)** | 0.35 | `ALPHA` | Peso do volume de consultas |
| **β (Beta)** | 0.40 | `BETA` | Peso da recência |
| **γ (Gamma)** | 0.25 | `GAMMA` | Peso dos sinais de negócio (billing, inquiry, views) |
| **λ (Lambda Decay)** | 0.15 | `LAMBDA_DECAY` | Taxa de decaimento exponencial. Maior λ = memória mais curta |

### Pesos do Modelo Híbrido

| Parâmetro | Valor | Env Var | Descrição |
|-----------|-------|---------|-----------|
| **W_MF** | 0.45 | `W_MF` | Peso do score do Matrix Factorization |
| **W_BIZ** | 0.55 | `W_BIZ` | Peso do score de negócio heurístico |

### Parâmetros do SVD

| Parâmetro | Valor | Env Var | Descrição |
|-----------|-------|---------|-----------|
| **N_COMPONENTS** | 50 | `N_COMPONENTS` | Número de dimensões latentes |
| **N_ITER** | 15 | `N_ITER` | Iterações do power method |
| **random_state** | 42 | (fixo) | Semente para reprodutibilidade |

### Limites de Output (Top-N)

| Parâmetro | Valor | Env Var | Descrição |
|-----------|-------|---------|-----------|
| **TOP_N_REPORTS** | 20 | `TOP_N_REPORTS` | Relatórios mais relevantes para warmup |
| **TOP_N_FEATURES** | 30 | `TOP_N_FEATURES` | Features mais consultadas |
| **TOP_N_CUSTOMERS** | 500 | `TOP_N_CUSTOMERS` | Clientes mais ativos para priorizar |
| **TOP_N_CDOCS** | 1000 | `TOP_N_CDOCS` | Documentos consultados mais frequentes |
| **TOP_N_PAIRS** | 2000 | `TOP_N_PAIRS` | Pares (customer→document) para warmup efetivo |

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

## 12. Configuração das Lambdas

### Lambda 1 — Trigger (`lambdas/trigger.py`)

Responsabilidade: detectar novo CSV no S3 e iniciar o SageMaker Processing Job.

```
Evento S3 (ObjectCreated)
    └─► Extrai bucket + key do evento
    └─► Valida extensão .csv
    └─► Chama sm.create_processing_job(...)
    └─► Retorna nome do job
```

### Lambda 2 — Warmup (`lambdas/warmup.py`)

Responsabilidade: ler o JSON de recomendações e chamar as APIs para popular o cache.

```
Evento S3 (ObjectCreated: warmup_recommendations.json)
    └─► Lê o JSON do S3
    └─► Itera top_pairs (limitado por MAX_PAIRS)
    └─► Para cada par: GET /reports/{reportName}?customer=X&consulted=Y
    └─► Contabiliza ok/erro e loga no CloudWatch
```

### Lambda 3 — Run Job (`lambdas/run_job.py`)

Serve dual purpose: CLI script (`__main__`) e Lambda handler. Permite execução manual ou programática do Processing Job.

**Recomendações para produção:**

- Use **AWS Secrets Manager** para a `API_KEY` em vez de variável de ambiente
- Adicione **retry com backoff exponencial** nas chamadas de API
- Use **SQS** como buffer entre o JSON de recomendações e as chamadas de API, para não ultrapassar o timeout da Lambda em volumes altos
- Implemente **dead-letter queue (DLQ)** para capturar falhas

---

## 13. Infraestrutura AWS

### Recursos Provisionados

| Recurso | Nome | Propósito |
|---------|------|-----------|
| **S3 Bucket** | `s3-bucket-time-7` | Armazena input (CSV), output (JSON) e código |
| **IAM Role (SM)** | `SageMakerExecutionRole` | Permite ao SageMaker acessar S3, CloudWatch, ECR |
| **IAM Role (Lambda)** | `CacheWarmupLambdaRole` | Permite Lambdas criar jobs, ler S3, escrever logs |
| **Lambda Trigger** | `cache-warmup-trigger` | Detecta CSV no S3 → inicia Processing Job |
| **Lambda Warmup** | `cache-warmup-warmup` | Detecta JSON no S3 → chama APIs de warmup |
| **SageMaker Processing Job** | `cache-warmup-YYYYMMDD-HHMMSS` | Executa o modelo ML (efêmero, sob demanda) |

### Layout do S3

```
s3-bucket-time-7/
├── input/                    # CSVs de logs (trigger automático)
├── output/warmup/            # JSON de recomendações (output do modelo)
└── code/
    └── recommender_ml.py     # Script executado pelo SageMaker
```

### Instância SageMaker

| Atributo | Valor |
|----------|-------|
| **Tipo** | `ml.m5.xlarge` |
| **vCPUs** | 4 |
| **RAM** | 16 GB |
| **Disco** | 20 GB EBS |
| **Imagem** | `sagemaker-scikit-learn:1.2-1-cpu-py3` |
| **Tempo máximo** | 3600s (1 hora) |
| **Custo estimado** | ~$0.23/hora (~$0.05-$0.50 por execução) |

---

## 14. Monitoramento e Operação

### CloudWatch — Métricas a acompanhar

| Métrica | Como verificar |
|---|---|
| Duração do Processing Job | SageMaker → Processing Jobs → Duration |
| Erros da Lambda de warmup | CloudWatch → Log Groups → `/aws/lambda/cache-warmup-executor` |
| Taxa de cache hit | Depende da implementação do cache (ElastiCache metrics) |
| Volume de CSVs processados | S3 → Metrics → NumberOfObjects |

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
---

## 16. Decisões de Design

### Por que Processing Job e não Endpoint?

| Aspecto | Processing Job (atual) | Endpoint |
|---------|----------------------|----------|
| **Uso** | Batch periódico | Inferência em tempo real |
| **Custo** | Paga só quando roda (~$0.50/exec) | Paga 24/7 (~$50+/mês) |
| **Latência** | 3-10 min total | < 100ms por predição |
| **Complexidade** | Baixa (script Python) | Alta (model registry, deploy, monitoring) |
| **Para MVP/Hackathon** | ✅ Ideal | ❌ Overengineering |

### Por que SVD e não Deep Learning?

- **Simplicidade**: SVD funciona out-of-the-box para collaborative filtering
- **Velocidade**: Treina em segundos/minutos vs. horas para neural CF
- **Interpretabilidade**: Fatores latentes podem ser inspecionados
- **Dados limitados**: DL precisa de muito mais dados para superar SVD
- **Imagem managed**: sklearn já está disponível na imagem SageMaker

### Por que Modelo Híbrido?

Apenas collaborative filtering (SVD) sofre de:
- **Cold start**: Novos customers/documents sem histórico = score 0
- **Popularity bias**: Tende a recomendar sempre os mais populares

O score de negócio complementa com:
- **Recência**: Consultas recentes são mais relevantes que antigas
- **Sinais explícitos**: Billing/inquiry indicam consultas "sérias"
- **Latência**: Alta latência sugere relatórios pesados (mais valor no cache)

### Quando Evoluir para Endpoint (Tempo Real)

| Trigger | Descrição |
|---------|-----------|
| Apps pedindo score em tempo real | Outros serviços precisam consultar "qual o score desse par?" via API |
| Volume > 10M registros | Batch processing começa a ultrapassar SLA de execução |
| Necessidade de A/B testing | Comparar versões do modelo em produção |
| Multi-tenant | Múltiplos consumidores com SLAs diferentes |
| ROI justifica custo fixo | Economia gerada pelo cache > custo do endpoint |

---

## 17. Glossário de Termos

### Termos de Domínio

| Termo | Definição |
|-------|-----------|
| **PJ** | Pessoa Jurídica (empresa/CNPJ) |
| **PF** | Pessoa Física (indivíduo/CPF) |
| **customerDocument** | Documento do cliente que faz a consulta (CNPJ/CPF do contratante) |
| **consultedDocument** | Documento da entidade sendo consultada (CNPJ/CPF alvo) |
| **reportName** | Nome do relatório de crédito solicitado |
| **FEATURENAME** | Feature/módulo específico dentro de um relatório |
| **billing** | Indica se a consulta gerou cobrança (1) ou não (0) |
| **inquiry** | Indica se foi uma consulta formal registrada no bureau |
| **post_report_view** | Indica se o relatório foi efetivamente visualizado após gerado |
| **httpTime** | Tempo de resposta da API em milissegundos |
| **inclusionDate** | Data/hora em que o registro foi inserido nos logs |
| **Cache Warmup** | Processo de pré-popular um cache antes que requisições reais cheguem |
| **Fallback Cache** | Cache secundário usado quando o sistema primário está lento/indisponível |

### Termos Técnicos de ML

| Termo | Definição |
|-------|-----------|
| **SVD (Singular Value Decomposition)** | Decomposição de matriz em fatores latentes. Captura relações implícitas entre entidades |
| **TruncatedSVD** | Variante eficiente do SVD que calcula apenas os top-K componentes |
| **Matrix Factorization** | Técnica que decompõe uma matriz esparsa em duas matrizes densas menores |
| **Collaborative Filtering** | Abordagem de recomendação baseada em padrões de comportamento coletivo |
| **Implicit Feedback** | Sinais indiretos de preferência (cliques, acessos) vs. explícitos (ratings) |
| **Latent Factors** | Dimensões ocultas que explicam padrões nos dados (ex: "perfil de risco") |
| **User Factors** | Representação vetorial de cada customer no espaço latente |
| **Item Factors** | Representação vetorial de cada document no espaço latente |
| **Dot Product** | Produto escalar entre vetores — quanto maior, maior a afinidade prevista |
| **MinMaxScaler** | Normaliza valores para o intervalo [0, 1] |
| **CSR Matrix** | Compressed Sparse Row — formato eficiente para matrizes com muitos zeros |
| **log1p** | Transformação log(1 + x) — suaviza distribuições com cauda longa |

---

## 18. Resultados de Exemplo

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

## 19. Comandos Úteis

```bash
# Deploy da infraestrutura
./deploy.sh

# Testar pipeline completo
./test_pipeline.sh data/sample_input.csv

# Upload do script para S3
aws s3 cp src/recommender_ml.py s3://<BUCKET>/code/recommender_ml.py

# Upload manual para disparar o pipeline
aws s3 cp data/sample_input.csv s3://s3-bucket-time-7/input/dados.csv

# Executar o job manualmente
python lambdas/run_job.py

# Verificar status do último job
aws sagemaker list-processing-jobs \
  --sort-by CreationTime --sort-order Descending --max-results 1

# Verificar status de um job específico
aws sagemaker describe-processing-job \
  --processing-job-name <job-name> \
  --query "ProcessingJobStatus"

# Logs da Lambda Trigger
aws logs tail /aws/lambda/cache-warmup-trigger --follow

# Logs da Lambda Warmup
aws logs tail /aws/lambda/cache-warmup-warmup --follow

# Baixar output
aws s3 cp s3://s3-bucket-time-7/output/warmup/warmup_recommendations.json ./output.json

# Inspecionar output
python3 -m json.tool output.json | head -50
```

---

## 20. Estrutura de Arquivos do Repositório

```
cache-warmup-recommender/
├── src/
│   ├── recommender.py          # Core scoring model (heurístico puro)
│   └── recommender_ml.py       # Modelo ML híbrido (SVD + heurístico) — versão principal
├── lambdas/
│   ├── trigger.py              # S3 CSV upload → inicia SageMaker Job
│   ├── warmup.py               # Lê JSON → chama APIs de warmup
│   └── run_job.py              # Launcher manual / Lambda handler
├── iam/
│   ├── sagemaker_policy.json   # Policy do SageMaker
│   └── lambda_policy.json      # Policy das Lambdas
├── data/
│   └── sample_output.json      # Exemplo de output do modelo
├── dashboard/
│   └── index.html              # Dashboard visual (abrir no browser)
├── docs/
│   └── DOCUMENTACAO_COMPLETA.md # Documentação detalhada (legado)
├── tests/
│   ├── conftest.py             # Fixtures compartilhadas
│   ├── test_data_loading.py    # Testes de carga de dados
│   ├── test_matrix.py          # Testes da matriz de interações
│   ├── test_recommendations.py # Testes de recomendações
│   ├── test_scoring.py         # Testes de scoring
│   ├── test_svd.py             # Testes do SVD
│   ├── test_lambda_trigger.py  # Testes da Lambda trigger
│   └── test_run_job.py         # Testes do run_job
├── .github/
│   └── workflows/
│       └── main.yml            # CI/CD pipeline
├── deploy.sh                   # Script de deploy AWS (CLI)
├── test_pipeline.sh            # Script de teste E2E
├── pyproject.toml              # Configuração Python (testes)
└── README.md                   # Este arquivo
```

---
