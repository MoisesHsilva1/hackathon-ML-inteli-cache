# Implementation Plan: ML Recommendation Model

## Overview

Refactor the existing `recommender_ml.py` prototype into a production-ready, modular ML recommendation pipeline with proper error handling, update the Lambda trigger and run_job script for ML parameter support, and establish a pytest + Hypothesis test suite validating 16 correctness properties.

## Tasks

- [ ] 1. Set up test infrastructure and shared fixtures
  - [ ] 1.1 Create test directory structure and conftest with Hypothesis strategies
    - Create `tests/` directory with `conftest.py` containing shared fixtures
    - Implement custom Hypothesis strategies: `st_access_log_df()`, `st_sparse_interaction_matrix()`, `st_factor_matrices()`, `st_scoring_params()`
    - Add a `pytest.ini` or `pyproject.toml` with pytest configuration (testpaths, markers)
    - Install dependencies: `pytest`, `hypothesis`, `numpy`, `pandas`, `scipy`, `scikit-learn`
    - _Requirements: Testing Strategy (Design)_

- [ ] 2. Implement Module 1: Data Loading & Validation
  - [ ] 2.1 Refactor `load_data` and `prepare` functions with proper error handling
    - Ensure `load_data` raises `FileNotFoundError` with descriptive message when no CSVs exist
    - Ensure `prepare` converts types correctly: `inclusionDate` → datetime, boolean ints for billing/inquiry/post_report_view, numeric coercion for httpTime/httpStatus with NaN→0
    - Ensure string columns are stripped of whitespace and quotation marks
    - Ensure `delta_days` = (max_inclusionDate - row_inclusionDate).total_seconds() / 86400.0
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 15.4_

  - [ ]* 2.2 Write property tests for data loading (Properties 1–3)
    - **Property 1: Data loading concatenation preserves all records**
    - **Property 2: Type coercion and string cleaning in prepare**
    - **Property 3: Delta days temporal computation**
    - **Validates: Requirements 1.1, 1.3, 1.4, 1.5, 15.4**

  - [ ]* 2.3 Write unit tests for data loading edge cases
    - Test `load_data` raises `FileNotFoundError` on empty directory
    - Test `prepare` handles non-numeric httpTime values gracefully
    - Test string cleaning removes both whitespace and quotes
    - _Requirements: 1.2, 1.3, 1.4_

- [ ] 3. Implement Module 2: Interaction Matrix Construction
  - [ ] 3.1 Refactor `build_interaction_matrix` with type hints and logging
    - Ensure pair counting uses groupby on (customerDocument, consultedDocument)
    - Ensure log1p transformation is applied to raw counts
    - Ensure CSR matrix shape is (n_unique_customers × n_unique_documents)
    - Ensure bidirectional index mappings are correct and complete
    - Log matrix dimensions, non-zero count, and density percentage
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 3.2 Write property tests for interaction matrix (Properties 4–5)
    - **Property 4: Interaction matrix values equal log1p of raw counts**
    - **Property 5: Matrix shape and index mapping consistency**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [ ] 4. Implement Module 3: SVD Training & Evaluation
  - [ ] 4.1 Refactor `train_svd` and `evaluate_model` with component clamping and reproducibility
    - Ensure `n_components` is clamped to `min(n_components, m-1, n-1)` without error
    - Ensure `random_state=42` for reproducibility
    - Ensure `user_factors` shape is (m, actual_components) and `item_factors` shape is (n, actual_components)
    - Ensure `evaluate_model` computes RMSE only on non-zero entries
    - Ensure RMSE returns 0.0 when no non-zero entries exist in sample
    - Log training time, explained variance, and actual component count
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 15.1, 15.3_

  - [ ]* 4.2 Write property tests for SVD (Properties 6–8)
    - **Property 6: SVD output shapes respect component clamping**
    - **Property 7: SVD training reproducibility**
    - **Property 8: Reconstruction RMSE uses only non-zero entries**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.4, 15.1, 15.3**

- [ ] 5. Implement Module 4: Scoring (MF + Business + Hybrid)
  - [ ] 5.1 Refactor `get_mf_scores_for_pairs`, `score_business`, and `compute_hybrid_pairs`
    - Ensure MF score = dot product for known pairs, 0.0 for unknown identifiers
    - Ensure MinMaxScaler normalization to [0, 1] for MF scores
    - Ensure Business_Score formula: ALPHA * V_norm + BETA * exp(-LAMBDA_DECAY * delta_mean) + GAMMA * W_biz
    - Ensure W_biz = 0.30*billing + 0.25*inquiry + 0.25*view + 0.20*latency_norm
    - Ensure Hybrid_Score = W_MF * mf_score + W_BIZ * biz_score, with missing MF defaulting to 0.0
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4, 6.5, 15.2_

  - [ ]* 5.2 Write property tests for scoring (Properties 9–12)
    - **Property 9: MF score computation (known pairs = dot product, unknown = 0.0)**
    - **Property 10: MF score normalization bounds**
    - **Property 11: Business score formula correctness**
    - **Property 12: Hybrid score formula correctness**
    - **Validates: Requirements 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 6.4, 15.2**

- [ ] 6. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement Module 5: Recommendation Generation
  - [ ] 7.1 Refactor `to_recs` and `generate_recommendations` with proper output structure
    - Ensure each dimension is sorted descending by its respective score (Business_Score or Hybrid_Score)
    - Ensure output list length = min(TOP_N, total_items) per dimension
    - Ensure each entry contains `volume` (int) and `score` (float rounded to 4 decimals)
    - Ensure all 5 dimensions are present: top_reports, top_features, top_customers, top_consulted_documents, top_pairs
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 7.2 Write property tests for recommendations (Properties 13–15)
    - **Property 13: Recommendation dimensions sorted descending by score**
    - **Property 14: Top-N list length bounded by configuration**
    - **Property 15: Recommendation entries contain volume and rounded score**
    - **Validates: Requirements 6.5, 7.2, 7.3, 7.4, 7.5**

- [ ] 8. Implement Module 6: Pipeline Orchestration and Output JSON
  - [ ] 8.1 Refactor the `run()` function and output JSON assembly
    - Ensure output JSON includes all required top-level keys: generated_at, model_version, model_params, model_metrics, warmup_targets, stats
    - Ensure `model_version` is "2.0-hybrid-svd"
    - Ensure `model_metrics` includes: svd_explained_variance_pct, svd_reconstruction_rmse, matrix_shape, matrix_density_pct, n_interactions, training_time_seconds
    - Ensure `stats` includes: total_records, unique_customers, unique_consulted, unique_reports, unique_features, date_range
    - Ensure JSON written with UTF-8 encoding, ensure_ascii=False, indent=2
    - Ensure `generated_at` is valid UTC ISO 8601 format
    - Log pipeline stages: file count, record count, date range, matrix info, training metrics, completion time, output path, top-5 pairs
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 13.1, 13.2, 13.3, 13.4, 13.5, 14.1, 14.2, 14.3_

  - [ ]* 8.2 Write unit tests for output JSON structure
    - Test output JSON has all required top-level keys
    - Test model_metrics has required keys
    - Test stats has required keys
    - Test model_version is "2.0-hybrid-svd"
    - Test generated_at is valid ISO format
    - Test model_params includes all formula fields
    - _Requirements: 8.2, 8.3, 8.4, 14.1, 14.2, 14.3_

- [ ] 9. Update Lambda Trigger for ML parameters
  - [ ] 9.1 Update `handler/lambda_trigger.py` with ML entrypoint and parameters
    - Update `ContainerEntrypoint` from `recommender.py` to `recommender_ml.py`
    - Add `W_MF`, `W_BIZ`, `N_COMPONENTS`, `N_ITER` to the Environment dict
    - Update `CODE_S3_URI` env var default to reference `recommender_ml.py`
    - Maintain the existing `.csv` extension check (skip non-CSV files)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [ ]* 9.2 Write property test and unit tests for Lambda trigger
    - **Property 16: Lambda trigger skips non-CSV files**
    - Test that Lambda starts job on .csv file upload
    - Test that ML parameters are passed in Environment dict
    - **Validates: Requirements 10.1, 10.2, 10.5**

- [ ] 10. Update Run Job Script for ML parameters
  - [ ] 10.1 Update `run_job.py` with ML parameters and entrypoint
    - Add `W_MF`, `W_BIZ`, `N_COMPONENTS`, `N_ITER` to `MODEL_PARAMS` dict
    - Update `CODE_S3_URI` default to `s3://s3-bucket-time-7/code/recommender_ml.py`
    - Update `ContainerEntrypoint` to reference `recommender_ml.py`
    - Maintain backward-compatible `lambda_handler` interface
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [ ]* 10.2 Write unit tests for run_job.py updates
    - Test that MODEL_PARAMS includes ML parameters
    - Test that CODE_S3_URI references recommender_ml.py
    - Test that lambda_handler interface remains backward-compatible
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [ ] 11. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Integration wiring and configuration validation
  - [ ] 12.1 Validate configuration management and environment variable defaults
    - Verify all env var reads in `recommender_ml.py` match the documented defaults (N_COMPONENTS=50, N_ITER=15, W_MF=0.45, W_BIZ=0.55, ALPHA=0.35, BETA=0.40, GAMMA=0.25, LAMBDA_DECAY=0.15, all TOP_N_* values, SM_INPUT_DIR, SM_OUTPUT_DIR, CSV_SEP)
    - Ensure the `run_job.py` Environment dict and `lambda_trigger.py` Environment dict are consistent with `recommender_ml.py` expected variables
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [ ]* 12.2 Write integration tests for end-to-end pipeline
    - Test that output JSON is written to the correct path
    - Test JSON encoding is UTF-8 with no ASCII escaping
    - Test that lambda_handler backward compatibility is maintained
    - _Requirements: 8.1, 8.5, 11.4_

- [ ] 13. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document (16 properties total)
- Unit tests validate specific examples and edge cases
- The existing `recommender_ml.py` prototype provides most logic; tasks focus on refactoring for robustness, proper error handling, and testability
- Python 3.10+ is used throughout (matches SageMaker scikit-learn image)
- No build system — tests run via `pytest tests/` directly

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "3.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.2", "4.1"] },
    { "id": 3, "tasks": ["4.2", "5.1"] },
    { "id": 4, "tasks": ["5.2", "7.1"] },
    { "id": 5, "tasks": ["7.2", "8.1"] },
    { "id": 6, "tasks": ["8.2", "9.1", "10.1"] },
    { "id": 7, "tasks": ["9.2", "10.2", "12.1"] },
    { "id": 8, "tasks": ["12.2"] }
  ]
}
```
