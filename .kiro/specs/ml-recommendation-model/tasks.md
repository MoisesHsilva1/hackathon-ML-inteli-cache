# Implementation Plan: ML Recommendation Model

## Overview

Implement a production-ready hybrid ML recommendation pipeline in `src/recommender_ml.py` that runs inside a SageMaker Processing Job. The pipeline combines TruncatedSVD collaborative filtering with business heuristics across 6 modular stages, validated by 15 correctness properties via Hypothesis property-based tests.

## Tasks

- [x] 1. Set up test infrastructure and shared fixtures
  - [x] 1.1 Create test directory structure and conftest with Hypothesis strategies
    - Create `tests/conftest.py` with shared fixtures and custom Hypothesis strategies
    - Implement strategies: `st_access_log_df()`, `st_sparse_interaction_matrix()`, `st_factor_matrices()`, `st_scoring_params()`
    - Configure pytest in `pyproject.toml` (testpaths, markers, hypothesis profile)
    - _Requirements: Testing Strategy (Design)_

- [x] 2. Implement Module 1: Data Loading & Validation
  - [x] 2.1 Implement `load_data` and `prepare` functions
    - `load_data`: glob all `.csv` files from input directory, read with pipe separator, concatenate into single DataFrame
    - `load_data`: raise `FileNotFoundError` with descriptive message when no CSV files exist
    - `prepare`: convert `inclusionDate` → datetime, `billing`/`inquiry`/`post_report_view` → boolean int (0/1)
    - `prepare`: coerce `httpTime` and `httpStatus` to numeric with `errors="coerce"`, fill NaN → 0
    - `prepare`: strip whitespace and quotation marks from string columns (reportName, TYPE_REPORT, FEATURENAME, FEATURE_TYPE, customerDocument, consultedDocument)
    - `prepare`: compute `delta_days = (max_inclusionDate - row_inclusionDate).total_seconds() / 86400.0`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 13.4_

  - [ ] 2.2 Write property tests for data loading (Properties 1–3)
    - **Property 1: Data loading concatenation preserves all records**
    - **Property 2: Type coercion and string cleaning in prepare**
    - **Property 3: Delta days temporal computation**
    - **Validates: Requirements 1.1, 1.3, 1.4, 1.5, 13.4**

  - [ ]* 2.3 Write unit tests for data loading edge cases
    - Test `load_data` raises `FileNotFoundError` on empty directory
    - Test `prepare` handles non-numeric httpTime values gracefully (coerce + fill 0)
    - Test string cleaning removes both leading/trailing whitespace and quotes
    - _Requirements: 1.2, 1.3, 1.4_

- [x] 3. Implement Module 2: Interaction Matrix Construction
  - [x] 3.1 Implement `build_interaction_matrix` function
    - Count interactions per unique (customerDocument, consultedDocument) pair via groupby
    - Apply `log1p` transformation to raw frequency counts
    - Construct scipy CSR sparse matrix of shape (n_unique_customers × n_unique_documents)
    - Build bidirectional index mappings: `cust_to_idx` and `doc_to_idx`
    - Log matrix dimensions, non-zero entry count, and density percentage
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 3.2 Write property tests for interaction matrix (Properties 4–5)
    - **Property 4: Interaction matrix values equal log1p of raw counts**
    - **Property 5: Matrix shape and index mapping consistency**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [x] 4. Checkpoint — Verify Modules 1–2 pass all tests
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement Module 3: SVD Training & Evaluation
  - [x] 5.1 Implement `train_svd` function
    - Train TruncatedSVD with configurable `n_components` (default 50) and `n_iter` (default 15)
    - Clamp `n_components` to `min(n_components, m - 1, n - 1)` when it exceeds matrix limits
    - Use `random_state=42` for reproducibility
    - Produce `user_factors` of shape (m, actual_components) and `item_factors` of shape (n, actual_components)
    - Log training duration, explained variance percentage, and actual component count
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 13.1_

  - [x] 5.2 Implement `evaluate_model` function
    - Sample up to 5000 users from the interaction matrix
    - Compute reconstruction RMSE only on non-zero entries (avoid sparse zero bias)
    - Return 0.0 if no non-zero entries exist in the evaluation sample
    - Record metrics: explained_variance_pct, reconstruction_rmse, matrix_shape, density_pct, n_interactions, training_time_seconds
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 13.3_

  - [ ]* 5.3 Write property tests for SVD (Properties 6–8)
    - **Property 6: SVD output shapes respect component clamping**
    - **Property 7: SVD training reproducibility**
    - **Property 8: Reconstruction RMSE uses only non-zero entries**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.4, 13.1, 13.3**

- [x] 6. Implement Module 4: Scoring (MF + Business + Hybrid)
  - [x] 6.1 Implement `get_mf_scores_for_pairs` function
    - Compute dot product of user_factor · item_factor for known customer-document pairs
    - Assign MF_Score = 0.0 for unknown customer or document identifiers
    - Normalize all MF scores to [0, 1] using MinMaxScaler
    - Compute scores for all unique (customerDocument, consultedDocument) pairs in the dataset
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 13.2_

  - [x] 6.2 Implement `score_business` function
    - Formula: `biz_score = ALPHA * V_norm + BETA * exp(-LAMBDA_DECAY * delta_mean) + GAMMA * W_biz`
    - W_biz = `0.30 * billing_rate + 0.25 * inquiry_rate + 0.25 * post_view_rate + 0.20 * latency_norm`
    - Accept configurable parameters from environment variables with defaults
    - _Requirements: 6.2, 6.3_

  - [x] 6.3 Implement `compute_hybrid_pairs` function
    - Formula: `score_final = W_MF * mf_score + W_BIZ * biz_score`
    - Default 0.0 for MF_Score when pair is missing from MF merge
    - Sort all pairs by Hybrid_Score in descending order
    - _Requirements: 6.1, 6.4, 6.5_

  - [ ]* 6.4 Write property tests for scoring (Properties 9–12)
    - **Property 9: MF score computation (known pairs = dot product, unknown = 0.0)**
    - **Property 10: MF score normalization bounds**
    - **Property 11: Business score formula correctness**
    - **Property 12: Hybrid score formula correctness**
    - **Validates: Requirements 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 6.4, 13.2**

- [x] 7. Checkpoint — Verify Modules 3–4 pass all tests
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement Module 5: Recommendation Generation
  - [x] 8.1 Implement `to_recs` and `generate_recommendations` functions
    - Generate 5 dimensions: top_reports, top_features, top_customers, top_consulted_documents, top_pairs
    - top_reports/top_features/top_customers/top_consulted_documents ranked by Business_Score descending
    - top_pairs ranked by Hybrid_Score descending
    - Configurable top-N limits: TOP_N_REPORTS=20, TOP_N_FEATURES=30, TOP_N_CUSTOMERS=500, TOP_N_CDOCS=1000, TOP_N_PAIRS=2000
    - Each entry contains `volume` (int) and `score` (float rounded to 4 decimal places)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 8.2 Write property tests for recommendations (Properties 13–15)
    - **Property 13: Recommendation dimensions sorted descending by score**
    - **Property 14: Top-N list length bounded by configuration**
    - **Property 15: Recommendation entries contain volume and rounded score**
    - **Validates: Requirements 6.5, 7.2, 7.3, 7.4, 7.5**

- [ ] 9. Implement Module 6: Pipeline Orchestration and Output JSON
  - [x] 9.1 Implement the `run()` function and JSON output assembly
    - Execute full pipeline: load → matrix → SVD → evaluate → hybrid scores → recommendations → write JSON
    - Output JSON top-level keys: `generated_at`, `model_version`, `model_params`, `model_metrics`, `warmup_targets`, `stats`
    - `model_version` = "2.0-hybrid-svd"
    - `model_metrics`: svd_explained_variance_pct, svd_reconstruction_rmse, matrix_shape, matrix_density_pct, n_interactions, training_time_seconds
    - `stats`: total_records, unique_customers, unique_consulted, unique_reports, unique_features, date_range (from/to)
    - `model_params`: formula string, W_biz formula string, all numeric parameters
    - Write JSON to `/opt/ml/processing/output/warmup_recommendations.json` with UTF-8, ensure_ascii=False, indent=2
    - `generated_at` in UTC ISO 8601 format
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 12.1, 12.2, 12.3_

  - [x] 9.2 Implement logging at each pipeline stage
    - Log INFO: number of CSV files loaded, total record count, date range, period duration
    - Log matrix dimensions (customers × documents), non-zero interactions
    - Log SVD training time, explained variance, component count
    - Log total execution time, output path
    - Log top-5 recommended pairs with customer, document, score, volume
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [x] 9.3 Implement configuration management (environment variable reads)
    - SVD: N_COMPONENTS=50, N_ITER=15
    - Hybrid weights: W_MF=0.45, W_BIZ=0.55
    - Business score: ALPHA=0.35, BETA=0.40, GAMMA=0.25, LAMBDA_DECAY=0.15
    - Top-N: TOP_N_REPORTS=20, TOP_N_FEATURES=30, TOP_N_CUSTOMERS=500, TOP_N_CDOCS=1000, TOP_N_PAIRS=2000
    - I/O paths: SM_INPUT_DIR=/opt/ml/processing/input, SM_OUTPUT_DIR=/opt/ml/processing/output, CSV_SEP=|
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [ ]* 9.4 Write unit tests for output JSON structure and config defaults
    - Test output JSON has all required top-level keys
    - Test model_metrics has required keys
    - Test stats has required keys
    - Test model_version is "2.0-hybrid-svd"
    - Test generated_at is valid ISO 8601 format
    - Test model_params includes all formula fields
    - Test all default config values match documented defaults
    - _Requirements: 8.2, 8.3, 8.4, 10.1, 10.2, 10.3, 10.4, 10.5, 12.1, 12.2, 12.3_

- [x] 10. Checkpoint — Verify Modules 5–6 pass all tests
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Integration tests for end-to-end pipeline
  - [x] 11.1 Write integration test for full pipeline execution
    - Create minimal sample CSV input, run `run()`, verify output JSON is written to correct path
    - Verify JSON encoding is UTF-8 with ensure_ascii=False
    - Verify output contains all 5 warmup_targets dimensions
    - Verify logging output covers all pipeline stages
    - _Requirements: 8.1, 8.5, 9.2, 9.3, 11.1, 11.2, 11.3, 11.4, 11.5_

- [x] 12. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate the 15 universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The existing `recommender_ml.py` prototype provides most logic; tasks focus on refactoring for robustness, proper error handling, and testability
- Python 3.10+ is used throughout (matches SageMaker scikit-learn:1.2-1 image)
- All dependencies (pandas, numpy, scipy, scikit-learn) are pre-installed in the SageMaker image
- Tests run via `pytest tests/` directly — no build system needed
- Scope is limited to `src/recommender_ml.py` execution inside the SageMaker Processing Job — no Lambda or run_job tasks

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.1", "3.1"] },
    { "id": 1, "tasks": ["2.2", "2.3", "3.2"] },
    { "id": 2, "tasks": ["5.1", "5.2"] },
    { "id": 3, "tasks": ["5.3", "6.1", "6.2"] },
    { "id": 4, "tasks": ["6.3", "6.4"] },
    { "id": 5, "tasks": ["8.1"] },
    { "id": 6, "tasks": ["8.2", "9.1", "9.2", "9.3"] },
    { "id": 7, "tasks": ["9.4", "11.1"] }
  ]
}
```
