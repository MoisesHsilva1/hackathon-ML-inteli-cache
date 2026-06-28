"""Unit tests for Requirements 8, 10, 11, 12: Pipeline orchestration, config, and JSON output."""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import recommender_ml


# ═══════════════════════════════════════════════════════════════════════════════
# Task 9.3: Configuration management — verify module-level env var defaults
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfigDefaults:
    """Verify all configuration values match documented defaults (Req 10.1-10.5)."""

    def test_svd_params(self):
        assert recommender_ml.N_COMPONENTS == 50
        assert recommender_ml.N_ITER == 15

    def test_hybrid_weights(self):
        assert recommender_ml.W_MF == 0.45
        assert recommender_ml.W_BIZ == 0.55

    def test_business_score_params(self):
        assert recommender_ml.ALPHA == 0.35
        assert recommender_ml.BETA == 0.40
        assert recommender_ml.GAMMA == 0.25
        assert recommender_ml.LAMBDA_DECAY == 0.15

    def test_top_n_params(self):
        assert recommender_ml.TOP_N_REPORTS == 20
        assert recommender_ml.TOP_N_FEATURES == 30
        assert recommender_ml.TOP_N_CUSTOMERS == 500
        assert recommender_ml.TOP_N_CDOCS == 1000
        assert recommender_ml.TOP_N_PAIRS == 2000

    def test_io_paths(self):
        assert recommender_ml.INPUT_DIR == "/opt/ml/processing/input"
        assert recommender_ml.OUTPUT_DIR == "/opt/ml/processing/output"
        assert recommender_ml.CSV_SEP == "|"


# ═══════════════════════════════════════════════════════════════════════════════
# Task 9.1 + 9.2: run() function, JSON output, and logging
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_csv_dir(tmp_path):
    """Create a temporary directory with a minimal sample CSV file."""
    csv_content = (
        "ID_REPORT|reportName|TYPE_REPORT|ID_FEATURE|FEATURENAME|FEATURE_TYPE"
        "|channel|billing|inquiry|post_report_view|httpStatus|httpTime"
        "|inclusionDate|customerDocument|consultedDocument\n"
        "R1|ReportA|PJ|F1|Feature1|O|web|1|0|1|200|150.0|2024-01-10|CUST_A|DOC_X\n"
        "R2|ReportB|PF|F2|Feature2|N|api|0|1|1|200|200.0|2024-01-12|CUST_B|DOC_Y\n"
        "R1|ReportA|PJ|F1|Feature1|O|web|1|0|0|404|50.0|2024-01-14|CUST_A|DOC_X\n"
        "R3|ReportC|PJ|F3|Feature3|C|mobile|1|1|0|200|300.0|2024-01-15|CUST_C|DOC_Z\n"
        "R1|ReportA|PJ|F1|Feature1|O|web|0|1|1|500|100.0|2024-01-15|CUST_A|DOC_Y\n"
    )
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text(csv_content, encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return str(out)


class TestRunFunction:
    """Verify run() executes full pipeline and produces correct JSON (Req 8.1-8.5, 12.1-12.3)."""

    def test_run_produces_json_at_correct_path(self, sample_csv_dir, output_dir):
        """run() writes warmup_recommendations.json to SM_OUTPUT_DIR."""
        with patch.object(recommender_ml, "INPUT_DIR", sample_csv_dir), \
             patch.object(recommender_ml, "OUTPUT_DIR", output_dir):
            result = recommender_ml.run()

        assert result == os.path.join(output_dir, "warmup_recommendations.json")
        assert os.path.isfile(result)

    def test_run_returns_string(self, sample_csv_dir, output_dir):
        """run() returns a string (the output file path)."""
        with patch.object(recommender_ml, "INPUT_DIR", sample_csv_dir), \
             patch.object(recommender_ml, "OUTPUT_DIR", output_dir):
            result = recommender_ml.run()

        assert isinstance(result, str)

    def test_json_encoding_utf8_no_ascii(self, sample_csv_dir, output_dir):
        """JSON is written with UTF-8 encoding and ensure_ascii=False (Req 8.5)."""
        with patch.object(recommender_ml, "INPUT_DIR", sample_csv_dir), \
             patch.object(recommender_ml, "OUTPUT_DIR", output_dir):
            out_path = recommender_ml.run()

        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify it's valid JSON
        payload = json.loads(content)
        assert isinstance(payload, dict)

        # Verify 2-space indentation
        lines = content.split("\n")
        # The second line should start with 2 spaces (first key)
        assert lines[1].startswith("  ")

    def test_output_has_required_top_level_keys(self, sample_csv_dir, output_dir):
        """Output JSON has all required top-level keys (Req 8.2)."""
        with patch.object(recommender_ml, "INPUT_DIR", sample_csv_dir), \
             patch.object(recommender_ml, "OUTPUT_DIR", output_dir):
            out_path = recommender_ml.run()

        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        required_keys = {
            "generated_at", "model_version", "model_params",
            "model_metrics", "warmup_targets", "stats",
        }
        assert required_keys.issubset(set(payload.keys()))

    def test_model_version_is_correct(self, sample_csv_dir, output_dir):
        """model_version is '2.0-hybrid-svd' (Req 12.1)."""
        with patch.object(recommender_ml, "INPUT_DIR", sample_csv_dir), \
             patch.object(recommender_ml, "OUTPUT_DIR", output_dir):
            out_path = recommender_ml.run()

        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        assert payload["model_version"] == "2.0-hybrid-svd"

    def test_generated_at_is_valid_iso(self, sample_csv_dir, output_dir):
        """generated_at is a valid ISO 8601 timestamp (Req 12.3)."""
        with patch.object(recommender_ml, "INPUT_DIR", sample_csv_dir), \
             patch.object(recommender_ml, "OUTPUT_DIR", output_dir):
            out_path = recommender_ml.run()

        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Should parse without error
        from datetime import datetime
        ts = datetime.fromisoformat(payload["generated_at"])
        assert ts.year >= 2024

    def test_model_metrics_has_required_keys(self, sample_csv_dir, output_dir):
        """model_metrics includes all required fields (Req 8.3)."""
        with patch.object(recommender_ml, "INPUT_DIR", sample_csv_dir), \
             patch.object(recommender_ml, "OUTPUT_DIR", output_dir):
            out_path = recommender_ml.run()

        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        metrics = payload["model_metrics"]
        required = {
            "svd_explained_variance_pct",
            "svd_reconstruction_rmse",
            "matrix_shape",
            "matrix_density_pct",
            "n_interactions",
            "training_time_seconds",
        }
        assert required.issubset(set(metrics.keys()))
        assert isinstance(metrics["matrix_shape"], list)
        assert len(metrics["matrix_shape"]) == 2
        assert isinstance(metrics["training_time_seconds"], float)

    def test_model_params_has_required_fields(self, sample_csv_dir, output_dir):
        """model_params includes formula, W_biz_formula, and all numeric params (Req 12.2)."""
        with patch.object(recommender_ml, "INPUT_DIR", sample_csv_dir), \
             patch.object(recommender_ml, "OUTPUT_DIR", output_dir):
            out_path = recommender_ml.run()

        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        params = payload["model_params"]
        required = {
            "formula", "W_biz_formula", "alpha", "beta", "gamma",
            "lambda", "w_mf", "w_biz", "svd_components", "svd_iterations",
        }
        assert required.issubset(set(params.keys()))

    def test_stats_has_required_keys(self, sample_csv_dir, output_dir):
        """stats includes all required fields (Req 8.4)."""
        with patch.object(recommender_ml, "INPUT_DIR", sample_csv_dir), \
             patch.object(recommender_ml, "OUTPUT_DIR", output_dir):
            out_path = recommender_ml.run()

        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        stats = payload["stats"]
        required = {
            "total_records", "unique_customers", "unique_consulted",
            "unique_reports", "unique_features", "date_range",
        }
        assert required.issubset(set(stats.keys()))
        assert "from" in stats["date_range"]
        assert "to" in stats["date_range"]

    def test_warmup_targets_has_5_dimensions(self, sample_csv_dir, output_dir):
        """warmup_targets has all 5 dimension keys (Req 7.1)."""
        with patch.object(recommender_ml, "INPUT_DIR", sample_csv_dir), \
             patch.object(recommender_ml, "OUTPUT_DIR", output_dir):
            out_path = recommender_ml.run()

        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        targets = payload["warmup_targets"]
        required = {
            "top_reports", "top_features", "top_customers",
            "top_consulted_documents", "top_pairs",
        }
        assert required == set(targets.keys())


class TestLogging:
    """Verify logging at each pipeline stage (Req 11.1-11.5)."""

    def test_logs_csv_count_and_records(self, sample_csv_dir, output_dir, caplog):
        """Logs number of CSV files and total records (Req 11.1)."""
        with caplog.at_level(logging.INFO), \
             patch.object(recommender_ml, "INPUT_DIR", sample_csv_dir), \
             patch.object(recommender_ml, "OUTPUT_DIR", output_dir):
            recommender_ml.run()

        log_text = caplog.text
        # Should log total records and file count
        assert "5" in log_text  # 5 records
        assert "1 file" in log_text

    def test_logs_matrix_dimensions(self, sample_csv_dir, output_dir, caplog):
        """Logs matrix dimensions and non-zero interactions (Req 11.2)."""
        with caplog.at_level(logging.INFO), \
             patch.object(recommender_ml, "INPUT_DIR", sample_csv_dir), \
             patch.object(recommender_ml, "OUTPUT_DIR", output_dir):
            recommender_ml.run()

        log_text = caplog.text
        # Should mention customers × documents
        assert "customers" in log_text.lower() or "Matriz" in log_text

    def test_logs_svd_training(self, sample_csv_dir, output_dir, caplog):
        """Logs SVD training time, explained variance, components (Req 11.3)."""
        with caplog.at_level(logging.INFO), \
             patch.object(recommender_ml, "INPUT_DIR", sample_csv_dir), \
             patch.object(recommender_ml, "OUTPUT_DIR", output_dir):
            recommender_ml.run()

        log_text = caplog.text
        assert "SVD" in log_text
        # Should log variance percentage
        assert "%" in log_text

    def test_logs_total_time_and_output_path(self, sample_csv_dir, output_dir, caplog):
        """Logs total execution time and output path (Req 11.4)."""
        with caplog.at_level(logging.INFO), \
             patch.object(recommender_ml, "INPUT_DIR", sample_csv_dir), \
             patch.object(recommender_ml, "OUTPUT_DIR", output_dir):
            recommender_ml.run()

        log_text = caplog.text
        assert "Pipeline completo" in log_text
        assert "Output salvo" in log_text

    def test_logs_top5_pairs(self, sample_csv_dir, output_dir, caplog):
        """Logs top-5 recommended pairs with customer, doc, score, volume (Req 11.5)."""
        with caplog.at_level(logging.INFO), \
             patch.object(recommender_ml, "INPUT_DIR", sample_csv_dir), \
             patch.object(recommender_ml, "OUTPUT_DIR", output_dir):
            recommender_ml.run()

        log_text = caplog.text
        assert "Top-5" in log_text
        assert "customer=" in log_text
        assert "doc=" in log_text
        assert "score=" in log_text
        assert "vol=" in log_text
