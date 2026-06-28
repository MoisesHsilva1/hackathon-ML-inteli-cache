"""Properties 1-3 + unit tests for Requirement 1: Data Loading & Validation."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import st_access_log_df
from src.recommender_ml import load_data, prepare


# ═══════════════════════════════════════════════════════════════════════════════
# Property 1: Data loading concatenation preserves all records
# Feature: ml-recommendation-model, Property 1: Data loading concatenation preserves all records
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.property
@settings(max_examples=100)
@given(
    n_files=st.integers(min_value=1, max_value=4),
    data=st.data(),
)
def test_load_data_concatenation_preserves_all_records(n_files, data):
    """**Validates: Requirements 1.1**

    For any set of N valid pipe-separated CSV files, load_data produces a
    DataFrame whose row count equals the sum of individual file row counts.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        total_rows = 0
        for i in range(n_files):
            df = data.draw(st_access_log_df(min_rows=2, max_rows=20))
            total_rows += len(df)
            path = os.path.join(tmpdir, f"data_{i:03d}.csv")
            df.to_csv(path, sep="|", index=False)

        result = load_data(tmpdir)
        assert len(result) == total_rows


# ═══════════════════════════════════════════════════════════════════════════════
# Property 2: Type coercion and string cleaning in prepare
# Feature: ml-recommendation-model, Property 2: Type coercion and string cleaning in prepare
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.property
@settings(max_examples=100)
@given(df=st_access_log_df(min_rows=3, max_rows=30))
def test_type_coercion_and_string_cleaning_in_prepare(df):
    """**Validates: Requirements 1.3, 1.4, 13.4**

    For any DataFrame with valid columns, after prepare:
    (a) billing, inquiry, post_report_view contain only 0 or 1;
    (b) httpTime and httpStatus contain no NaN;
    (c) string columns have no leading/trailing whitespace or quotation marks.
    """
    prepared, t_ref = prepare(df)

    # (a) Boolean int columns contain only 0 or 1
    for col in ("billing", "inquiry", "post_report_view"):
        unique_vals = set(prepared[col].unique())
        assert unique_vals <= {0, 1}, (
            f"Column '{col}' contains values other than 0/1: {unique_vals}"
        )

    # (b) httpTime and httpStatus contain no NaN
    assert not prepared["httpTime"].isna().any(), "httpTime contains NaN after prepare"
    assert not prepared["httpStatus"].isna().any(), "httpStatus contains NaN after prepare"

    # (c) String columns have no leading/trailing whitespace or quotation marks
    str_cols = [
        "reportName",
        "TYPE_REPORT",
        "FEATURENAME",
        "FEATURE_TYPE",
        "customerDocument",
        "consultedDocument",
    ]
    for col in str_cols:
        for val in prepared[col]:
            s = str(val)
            assert s == s.strip(), (
                f"Column '{col}' has leading/trailing whitespace: '{s}'"
            )
            assert not s.startswith('"') and not s.endswith('"'), (
                f"Column '{col}' has quotation marks: '{s}'"
            )
            assert not s.startswith("'") and not s.endswith("'"), (
                f"Column '{col}' has single quotes: '{s}'"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 3: Delta days temporal computation
# Feature: ml-recommendation-model, Property 3: Delta days temporal computation
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.property
@settings(max_examples=100)
@given(df=st_access_log_df(min_rows=3, max_rows=30))
def test_delta_days_temporal_computation(df):
    """**Validates: Requirements 1.5**

    For any DataFrame with valid inclusionDate column, after prepare,
    delta_days = (max_inclusionDate - row_inclusionDate).total_seconds() / 86400.0,
    and the max row has delta_days == 0.0.
    """
    prepared, t_ref = prepare(df)

    # t_ref should be the max inclusionDate
    assert t_ref == prepared["inclusionDate"].max()

    # Verify delta_days computation for each row
    for idx, row in prepared.iterrows():
        expected = (t_ref - row["inclusionDate"]).total_seconds() / 86400.0
        assert np.isclose(row["delta_days"], expected, atol=1e-9), (
            f"Row {idx}: expected delta_days={expected}, got {row['delta_days']}"
        )

    # The row(s) with the maximum inclusionDate should have delta_days == 0.0
    max_date_mask = prepared["inclusionDate"] == t_ref
    assert (prepared.loc[max_date_mask, "delta_days"] == 0.0).all(), (
        "Rows with max inclusionDate should have delta_days == 0.0"
    )
