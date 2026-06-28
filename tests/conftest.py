"""Shared fixtures and Hypothesis strategies for the ML recommendation model tests."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import strategies as st
from scipy.sparse import csr_matrix

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════════════
# Custom Hypothesis Strategies
# ═══════════════════════════════════════════════════════════════════════════════


@st.composite
def st_access_log_df(
    draw: st.DrawFn,
    min_rows: int = 3,
    max_rows: int = 50,
) -> pd.DataFrame:
    """Generate random DataFrames with valid access log column schema.

    Columns: ID_REPORT, reportName, TYPE_REPORT, ID_FEATURE, FEATURENAME,
    FEATURE_TYPE, channel, billing, inquiry, post_report_view, httpStatus,
    httpTime, inclusionDate, customerDocument, consultedDocument
    """
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    id_reports = draw(
        st.lists(
            st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=3, max_size=8),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    report_names = draw(
        st.lists(
            st.sampled_from(["ReportA", "ReportB", "ReportC", " ReportD ", '"ReportE"']),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    type_reports = draw(
        st.lists(
            st.sampled_from(["PJ", "PF", " PJ ", '"PF"']),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    id_features = draw(
        st.lists(
            st.text(alphabet="0123456789", min_size=1, max_size=5),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    feature_names = draw(
        st.lists(
            st.sampled_from(["FeatureX", " FeatureY ", '"FeatureZ"', "Feature W"]),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    feature_types = draw(
        st.lists(
            st.sampled_from(["O", "N", "C", " O ", '"N"']),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    channels = draw(
        st.lists(
            st.sampled_from(["web", "api", "mobile"]),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    billing = draw(
        st.lists(
            st.sampled_from([0, 1, True, False, "1", "0"]),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    inquiry = draw(
        st.lists(
            st.sampled_from([0, 1, True, False, "1", "0"]),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    post_report_view = draw(
        st.lists(
            st.sampled_from([0, 1, True, False, "1", "0"]),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    http_status = draw(
        st.lists(
            st.sampled_from([200, 201, 400, 404, 500, "200", "invalid", None]),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    http_time = draw(
        st.lists(
            st.one_of(
                st.floats(min_value=0, max_value=5000, allow_nan=False, allow_infinity=False),
                st.just("invalid"),
                st.just(None),
            ),
            min_size=n_rows,
            max_size=n_rows,
        )
    )

    # Generate inclusion dates within a 30-day window
    base_ts = pd.Timestamp("2024-01-01")
    offsets = draw(
        st.lists(
            st.integers(min_value=0, max_value=30 * 24 * 3600),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    inclusion_dates = [base_ts + pd.Timedelta(seconds=s) for s in offsets]

    # Customer and consulted documents (allow some repetition for pairs)
    n_customers = draw(st.integers(min_value=2, max_value=min(10, n_rows)))
    n_documents = draw(st.integers(min_value=2, max_value=min(10, n_rows)))
    customer_pool = [f"CUST_{i:04d}" for i in range(n_customers)]
    document_pool = [f"DOC_{i:04d}" for i in range(n_documents)]

    customer_docs = draw(
        st.lists(
            st.sampled_from(customer_pool),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    consulted_docs = draw(
        st.lists(
            st.sampled_from(document_pool),
            min_size=n_rows,
            max_size=n_rows,
        )
    )

    df = pd.DataFrame(
        {
            "ID_REPORT": id_reports,
            "reportName": report_names,
            "TYPE_REPORT": type_reports,
            "ID_FEATURE": id_features,
            "FEATURENAME": feature_names,
            "FEATURE_TYPE": feature_types,
            "channel": channels,
            "billing": billing,
            "inquiry": inquiry,
            "post_report_view": post_report_view,
            "httpStatus": http_status,
            "httpTime": http_time,
            "inclusionDate": inclusion_dates,
            "customerDocument": customer_docs,
            "consultedDocument": consulted_docs,
        }
    )
    return df


@st.composite
def st_sparse_interaction_matrix(
    draw: st.DrawFn,
    min_rows: int = 3,
    max_rows: int = 50,
    min_cols: int = 3,
    max_cols: int = 50,
    min_density: float = 0.01,
    max_density: float = 0.5,
) -> csr_matrix:
    """Generate random CSR sparse matrices of varying density.

    Values represent log1p-transformed counts (non-negative floats).
    """
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    density = draw(st.floats(min_value=min_density, max_value=max_density))

    n_elements = max(1, int(n_rows * n_cols * density))
    rows = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_rows - 1),
            min_size=n_elements,
            max_size=n_elements,
        )
    )
    cols = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_cols - 1),
            min_size=n_elements,
            max_size=n_elements,
        )
    )
    # Values are log1p of counts (always >= 0)
    raw_counts = draw(
        st.lists(
            st.integers(min_value=1, max_value=100),
            min_size=n_elements,
            max_size=n_elements,
        )
    )
    values = np.log1p(np.array(raw_counts, dtype=np.float32))

    matrix = csr_matrix(
        (values, (np.array(rows), np.array(cols))),
        shape=(n_rows, n_cols),
    )
    return matrix


@st.composite
def st_factor_matrices(
    draw: st.DrawFn,
    min_users: int = 3,
    max_users: int = 30,
    min_items: int = 3,
    max_items: int = 30,
    min_components: int = 2,
    max_components: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random user/item factor matrix pairs.

    Returns (user_factors, item_factors) with matching component dimension.
    """
    n_users = draw(st.integers(min_value=min_users, max_value=max_users))
    n_items = draw(st.integers(min_value=min_items, max_value=max_items))
    n_components = draw(st.integers(min_value=min_components, max_value=max_components))

    user_factors = np.array(
        draw(
            st.lists(
                st.lists(
                    st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
                    min_size=n_components,
                    max_size=n_components,
                ),
                min_size=n_users,
                max_size=n_users,
            )
        ),
        dtype=np.float64,
    )
    item_factors = np.array(
        draw(
            st.lists(
                st.lists(
                    st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
                    min_size=n_components,
                    max_size=n_components,
                ),
                min_size=n_items,
                max_size=n_items,
            )
        ),
        dtype=np.float64,
    )
    return user_factors, item_factors


@st.composite
def st_scoring_params(draw: st.DrawFn) -> tuple[float, float, float, float]:
    """Generate valid (ALPHA, BETA, GAMMA, LAMBDA_DECAY) tuples.

    All values are non-negative. ALPHA + BETA + GAMMA can sum to any positive value
    (the formula does not require normalization to 1.0).
    """
    alpha = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    beta = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    gamma = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    lambda_decay = draw(
        st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    return alpha, beta, gamma, lambda_decay


# ═══════════════════════════════════════════════════════════════════════════════
# Shared Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_access_log_df() -> pd.DataFrame:
    """A small deterministic DataFrame for unit tests."""
    return pd.DataFrame(
        {
            "ID_REPORT": ["R001", "R002", "R001", "R003", "R001"],
            "reportName": ["Report A", "Report B", "Report A", "Report C", "Report A"],
            "TYPE_REPORT": ["PJ", "PF", "PJ", "PJ", "PJ"],
            "ID_FEATURE": ["F01", "F02", "F01", "F03", "F01"],
            "FEATURENAME": ["Feature1", "Feature2", "Feature1", "Feature3", "Feature1"],
            "FEATURE_TYPE": ["O", "N", "O", "C", "O"],
            "channel": ["web", "api", "web", "mobile", "web"],
            "billing": [1, 0, 1, 1, 0],
            "inquiry": [0, 1, 0, 1, 1],
            "post_report_view": [1, 1, 0, 0, 1],
            "httpStatus": [200, 200, 404, 200, 500],
            "httpTime": [150.0, 200.0, 50.0, 300.0, 100.0],
            "inclusionDate": pd.to_datetime(
                [
                    "2024-01-10",
                    "2024-01-12",
                    "2024-01-14",
                    "2024-01-15",
                    "2024-01-15",
                ]
            ),
            "customerDocument": ["CUST_A", "CUST_B", "CUST_A", "CUST_C", "CUST_A"],
            "consultedDocument": ["DOC_X", "DOC_Y", "DOC_X", "DOC_Z", "DOC_Y"],
        }
    )


@pytest.fixture
def sample_interaction_matrix() -> tuple[csr_matrix, dict, dict]:
    """A small deterministic interaction matrix for unit tests."""
    # 3 customers × 3 documents, with known counts
    data = np.log1p(np.array([3, 1, 2, 1], dtype=np.float32))
    rows = np.array([0, 0, 1, 2])
    cols = np.array([0, 1, 2, 0])
    matrix = csr_matrix((data, (rows, cols)), shape=(3, 3))
    cust_to_idx = {"CUST_A": 0, "CUST_B": 1, "CUST_C": 2}
    doc_to_idx = {"DOC_X": 0, "DOC_Y": 1, "DOC_Z": 2}
    return matrix, cust_to_idx, doc_to_idx
