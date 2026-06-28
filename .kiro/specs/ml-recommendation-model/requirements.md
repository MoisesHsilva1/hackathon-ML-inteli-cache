# Requirements Document

## Introduction

This feature implements a production-ready ML recommendation model that trains on the existing Polaris access log dataset (~724K records) and deploys as a SageMaker Processing Job. The model evolves the current heuristic-only recommender (v1) and hybrid prototype (v2) into a collaborative filtering system that combines Matrix Factorization (TruncatedSVD) with business heuristics to generate cache warmup recommendations.

The model builds a customer×document interaction matrix from implicit feedback (access frequency), trains a TruncatedSVD decomposition, and produces a hybrid score that blends ML predictions with the existing business scoring formula. Output is a structured JSON file written to the SageMaker output channel.

## Glossary

- **Recommender**: The Python script (`recommender_ml.py`) that runs inside the SageMaker Processing Job, responsible for loading data, training the model, and producing recommendation output
- **Interaction_Matrix**: A sparse matrix of shape (n_customers × n_documents) where cell values represent the log1p-transformed access frequency between a customer and a consulted document
- **SVD_Model**: A TruncatedSVD decomposition that factorizes the Interaction_Matrix into user factors and item factors of configurable dimensionality
- **MF_Score**: The Matrix Factorization score for a customer-document pair, computed as the dot product of the user factor vector and the item factor vector, normalized to [0, 1]
- **Business_Score**: The heuristic score combining volume, recency (exponential decay), and business weight (billing, inquiry, post_view, latency)
- **Hybrid_Score**: The final recommendation score combining MF_Score and Business_Score using configurable weights: score_final = W_MF · MF_Score + W_BIZ · Business_Score
- **Warmup_JSON**: The output JSON file containing top-N recommendations across 5 dimensions (reports, features, customers, documents, pairs) plus model metrics and dataset statistics
- **Processing_Job**: An AWS SageMaker Processing Job using the scikit-learn:1.2-1-cpu-py3 managed image on ml.m5.xlarge instance
- **Input_CSV**: A pipe-separated (|) CSV file containing Polaris access logs with fields: ID_REPORT, reportName, TYPE_REPORT, ID_FEATURE, FEATURENAME, FEATURE_TYPE, channel, billing, inquiry, post_report_view, httpStatus, httpTime, inclusionDate, customerDocument, consultedDocument
- **Implicit_Feedback**: Access frequency used as a proxy for user preference, transformed with log1p to reduce the effect of power-law distributions
- **Explained_Variance**: The fraction of the total variance in the Interaction_Matrix captured by the SVD_Model components
- **Reconstruction_RMSE**: The root mean square error between the original Interaction_Matrix values and the SVD_Model reconstruction for non-zero entries

## Requirements

### Requirement 1: Data Loading and Validation

**User Story:** As a data engineer, I want the Recommender to load and validate all CSV files from the SageMaker input channel, so that the model trains on complete and correctly typed data.

#### Acceptance Criteria

1. WHEN one or more CSV files exist in the `/opt/ml/processing/input` directory, THE Recommender SHALL load all files using pipe `|` as separator and concatenate them into a single DataFrame
2. WHEN no CSV files exist in the input directory, THE Recommender SHALL raise a FileNotFoundError with a descriptive message indicating the empty directory path
3. WHEN a CSV file is loaded, THE Recommender SHALL convert `inclusionDate` to datetime, `billing` to boolean integer (0 or 1), `inquiry` to boolean integer (0 or 1), `post_report_view` to boolean integer (0 or 1), `httpTime` to numeric with NaN filled as 0, and `httpStatus` to numeric with NaN filled as 0
4. WHEN string columns (reportName, TYPE_REPORT, FEATURENAME, FEATURE_TYPE, customerDocument, consultedDocument) are loaded, THE Recommender SHALL strip leading/trailing whitespace and quotation marks from all values
5. WHEN data preparation is complete, THE Recommender SHALL compute `delta_days` as the number of seconds between each record's inclusionDate and the maximum inclusionDate in the dataset, divided by 86400

### Requirement 2: Interaction Matrix Construction

**User Story:** As a data scientist, I want the Recommender to build a sparse customer×document interaction matrix using implicit feedback, so that the SVD_Model can learn latent preference patterns.

#### Acceptance Criteria

1. WHEN building the Interaction_Matrix, THE Recommender SHALL count the number of interactions per unique (customerDocument, consultedDocument) pair
2. WHEN raw interaction counts are computed, THE Recommender SHALL apply log1p transformation to the frequency values to reduce the dominance of power-law distributed heavy users
3. THE Recommender SHALL construct a scipy CSR sparse matrix of shape (n_unique_customers × n_unique_documents)
4. THE Recommender SHALL maintain bidirectional index mappings (customer_to_index, document_to_index) for translating between original identifiers and matrix positions
5. WHEN the Interaction_Matrix is constructed, THE Recommender SHALL log the matrix dimensions, number of non-zero entries, and matrix density percentage

### Requirement 3: SVD Model Training

**User Story:** As a data scientist, I want the Recommender to train a TruncatedSVD model with configurable hyperparameters, so that latent factors capture collaborative filtering signals.

#### Acceptance Criteria

1. THE Recommender SHALL train a TruncatedSVD model on the Interaction_Matrix using configurable N_COMPONENTS (default 50) and N_ITER (default 15) parameters read from environment variables
2. WHEN N_COMPONENTS exceeds (min(n_customers, n_documents) - 1), THE Recommender SHALL reduce N_COMPONENTS to the maximum allowable value without raising an error
3. WHEN training completes, THE SVD_Model SHALL produce user_factors of shape (n_customers, actual_components) and item_factors of shape (n_documents, actual_components)
4. THE Recommender SHALL use random_state=42 for reproducible SVD training results across identical inputs
5. WHEN training completes, THE Recommender SHALL log the training duration in seconds, the Explained_Variance as a percentage, and the actual number of components used

### Requirement 4: Model Evaluation

**User Story:** As a data scientist, I want the Recommender to evaluate model quality using reconstruction metrics, so that I can monitor model performance over time.

#### Acceptance Criteria

1. WHEN training completes, THE Recommender SHALL evaluate the SVD_Model by computing Reconstruction_RMSE on a random sample of up to 5000 users from the Interaction_Matrix
2. THE Recommender SHALL compute Reconstruction_RMSE only on non-zero entries of the original matrix to avoid bias from the sparse zero entries
3. THE Recommender SHALL record model metrics including: Explained_Variance percentage, Reconstruction_RMSE, matrix shape, matrix density percentage, number of non-zero interactions, and training time in seconds
4. WHEN the non-zero mask contains zero entries in the sample, THE Recommender SHALL report Reconstruction_RMSE as 0.0 instead of raising a division error

### Requirement 5: MF Score Computation

**User Story:** As a data scientist, I want the Recommender to compute Matrix Factorization scores for observed customer-document pairs, so that collaborative filtering predictions can be combined with business heuristics.

#### Acceptance Criteria

1. WHEN computing MF_Score for a customer-document pair, THE Recommender SHALL calculate the dot product of the customer's user_factor vector and the document's item_factor vector
2. WHEN a customer or document identifier is not present in the index mappings, THE Recommender SHALL assign an MF_Score of 0.0 for that pair
3. WHEN all raw MF scores are computed, THE Recommender SHALL normalize them to the range [0, 1] using MinMaxScaler
4. THE Recommender SHALL compute MF scores for all unique (customerDocument, consultedDocument) pairs observed in the dataset

### Requirement 6: Hybrid Score Calculation

**User Story:** As a data scientist, I want the Recommender to combine ML predictions with business heuristics into a single hybrid score, so that recommendations balance collaborative filtering signals with domain expertise.

#### Acceptance Criteria

1. THE Recommender SHALL compute Hybrid_Score using the formula: score_final = W_MF · MF_Score + W_BIZ · Business_Score, where W_MF (default 0.45) and W_BIZ (default 0.55) are configurable via environment variables
2. THE Recommender SHALL compute Business_Score using the formula: Business_Score = ALPHA · V_norm + BETA · exp(-LAMBDA_DECAY · delta_mean_days) + GAMMA · W_biz, where ALPHA (default 0.35), BETA (default 0.40), GAMMA (default 0.25), and LAMBDA_DECAY (default 0.15) are configurable via environment variables
3. THE Recommender SHALL compute W_biz as: 0.30 · billing_rate + 0.25 · inquiry_rate + 0.25 · post_view_rate + 0.20 · latency_norm
4. WHEN MF_Score is unavailable for a pair (missing from the merge), THE Recommender SHALL use 0.0 as the MF_Score contribution to the Hybrid_Score
5. THE Recommender SHALL sort all pairs by Hybrid_Score in descending order

### Requirement 7: Recommendation Generation

**User Story:** As a cache administrator, I want the Recommender to produce top-N recommendations across 5 dimensions, so that the output can be used to pre-populate the cache with the most valuable entries.

#### Acceptance Criteria

1. THE Recommender SHALL generate recommendations in 5 dimensions: top_reports (grouped by reportName, TYPE_REPORT), top_features (grouped by ID_FEATURE, FEATURENAME, FEATURE_TYPE), top_customers (grouped by customerDocument), top_consulted_documents (grouped by consultedDocument), and top_pairs (grouped by customerDocument, consultedDocument)
2. THE Recommender SHALL use configurable top-N limits via environment variables: TOP_N_REPORTS (default 20), TOP_N_FEATURES (default 30), TOP_N_CUSTOMERS (default 500), TOP_N_CDOCS (default 1000), TOP_N_PAIRS (default 2000)
3. WHEN generating top_reports, top_features, top_customers, and top_consulted_documents, THE Recommender SHALL rank items by Business_Score in descending order
4. WHEN generating top_pairs, THE Recommender SHALL rank pairs by Hybrid_Score in descending order
5. THE Recommender SHALL include volume (integer count) and score (float rounded to 4 decimal places) for each recommendation entry

### Requirement 8: Output JSON Structure

**User Story:** As a downstream consumer, I want the Recommender output to follow a well-defined JSON schema with model metadata, so that consumers can validate and interpret recommendations correctly.

#### Acceptance Criteria

1. THE Recommender SHALL produce a single JSON file named `warmup_recommendations.json` in the `/opt/ml/processing/output` directory
2. THE Recommender SHALL include in the output JSON the following top-level keys: `generated_at` (UTC ISO timestamp), `model_version` (string "2.0-hybrid-svd"), `model_params` (all formula parameters and weights), `model_metrics` (evaluation results), `warmup_targets` (5 recommendation dimensions), and `stats` (dataset summary statistics)
3. THE Recommender SHALL include in `model_metrics`: svd_explained_variance_pct, svd_reconstruction_rmse, matrix_shape, matrix_density_pct, n_interactions, and training_time_seconds
4. THE Recommender SHALL include in `stats`: total_records, unique_customers, unique_consulted, unique_reports, unique_features, and date_range (object with `from` and `to` date strings)
5. THE Recommender SHALL write the JSON file with UTF-8 encoding, ensure_ascii=False, and 2-space indentation

### Requirement 9: SageMaker Processing Job Execution

**User Story:** As a DevOps engineer, I want the Recommender to run as a SageMaker Processing Job with appropriate resource configuration, so that training scales with dataset size and integrates with the existing AWS pipeline.

#### Acceptance Criteria

1. THE Processing_Job SHALL use the AWS-managed scikit-learn:1.2-1-cpu-py3 container image on an ml.m5.xlarge instance with 20GB volume
2. THE Processing_Job SHALL read input data from the SageMaker input channel mounted at `/opt/ml/processing/input`
3. THE Processing_Job SHALL write output data to the SageMaker output channel mounted at `/opt/ml/processing/output` with S3UploadMode set to EndOfJob
4. THE Processing_Job SHALL accept all model hyperparameters (ALPHA, BETA, GAMMA, LAMBDA_DECAY, W_MF, W_BIZ, N_COMPONENTS, N_ITER, TOP_N_REPORTS, TOP_N_FEATURES, TOP_N_CUSTOMERS, TOP_N_CDOCS, TOP_N_PAIRS) via environment variables
5. THE Processing_Job SHALL complete execution within 3600 seconds (MaxRuntimeInSeconds) for datasets up to 1 million records

### Requirement 10: Configuration Management

**User Story:** As a data scientist, I want all model hyperparameters to be configurable via environment variables with sensible defaults, so that I can tune the model without code changes.

#### Acceptance Criteria

1. THE Recommender SHALL read the following SVD parameters from environment variables with defaults: N_COMPONENTS=50, N_ITER=15
2. THE Recommender SHALL read the following hybrid weights from environment variables with defaults: W_MF=0.45, W_BIZ=0.55
3. THE Recommender SHALL read the following business score parameters from environment variables with defaults: ALPHA=0.35, BETA=0.40, GAMMA=0.25, LAMBDA_DECAY=0.15
4. THE Recommender SHALL read the following top-N parameters from environment variables with defaults: TOP_N_REPORTS=20, TOP_N_FEATURES=30, TOP_N_CUSTOMERS=500, TOP_N_CDOCS=1000, TOP_N_PAIRS=2000
5. THE Recommender SHALL read I/O path parameters from environment variables with defaults: SM_INPUT_DIR=/opt/ml/processing/input, SM_OUTPUT_DIR=/opt/ml/processing/output, CSV_SEP=|

### Requirement 11: Logging and Observability

**User Story:** As a DevOps engineer, I want the Recommender to produce structured logs at each pipeline stage, so that I can monitor execution progress and diagnose failures via CloudWatch.

#### Acceptance Criteria

1. THE Recommender SHALL log at INFO level: the number of CSV files loaded, total record count, date range of the dataset, and period duration in days
2. WHEN the Interaction_Matrix is built, THE Recommender SHALL log the matrix dimensions (customers × documents) and the number of non-zero interactions
3. WHEN SVD training completes, THE Recommender SHALL log the training time, explained variance percentage, and actual component count
4. WHEN the pipeline completes, THE Recommender SHALL log the total execution time and the path where the output JSON was saved
5. WHEN the pipeline completes, THE Recommender SHALL log the top-5 recommended pairs with their customer identifier, document identifier, score value, and volume count

### Requirement 12: Model Versioning and Traceability

**User Story:** As a data scientist, I want the output to include the model version and complete parameter snapshot, so that I can reproduce results and track model evolution across runs.

#### Acceptance Criteria

1. THE Recommender SHALL include `model_version` as "2.0-hybrid-svd" in the output JSON to distinguish from the heuristic-only v1 output
2. THE Recommender SHALL include in `model_params` the complete scoring formula as a string, the W_biz sub-formula, and all numeric parameter values (alpha, beta, gamma, lambda, w_mf, w_biz, svd_components, svd_iterations)
3. THE Recommender SHALL include the `generated_at` timestamp in UTC ISO 8601 format representing the moment the pipeline produces the output

### Requirement 13: Error Resilience

**User Story:** As a DevOps engineer, I want the Recommender to handle edge cases gracefully, so that transient data issues do not cause complete pipeline failures.

#### Acceptance Criteria

1. IF the input dataset contains fewer unique customers or documents than N_COMPONENTS, THEN THE Recommender SHALL reduce N_COMPONENTS to (min(n_customers, n_documents) - 1) and proceed with training
2. IF a customer-document pair in the MF score computation references an identifier absent from the index mappings, THEN THE Recommender SHALL assign a score of 0.0 for that pair without raising an exception
3. IF the sampled users for evaluation contain no non-zero entries, THEN THE Recommender SHALL report Reconstruction_RMSE as 0.0
4. IF `httpTime` column contains non-numeric values, THEN THE Recommender SHALL coerce them to numeric and fill NaN with 0 before computing latency normalization
