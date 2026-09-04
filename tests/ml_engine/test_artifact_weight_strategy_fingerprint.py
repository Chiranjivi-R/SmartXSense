"""Regression tests for Bug 2: XGBoost sample_weight_strategy artifact fingerprinting."""

from __future__ import annotations

from ml_engine.models.artifact_store import config_fingerprint
from ml_engine.models.trainer import expected_configuration_for_request


def test_xgboost_sample_weight_strategy_fingerprint_difference() -> None:
    """Proves that sample_weight_strategy=None and sample_weight_strategy='spike_sensitive' do NOT share fingerprint."""
    config_none = expected_configuration_for_request(
        model_key="xgboost",
        domain="retail",
        sample_weight_strategy=None,
    )
    config_spike = expected_configuration_for_request(
        model_key="xgboost",
        domain="retail",
        sample_weight_strategy="spike_sensitive",
    )

    fp_none = config_fingerprint(config_none)
    fp_spike = config_fingerprint(config_spike)

    assert fp_none != fp_spike, "Configurations with different sample_weight_strategy MUST produce different fingerprints"


def test_expected_configuration_contains_sample_weight_strategy() -> None:
    """Verify that expected configuration returned for XGBoost contains sample_weight_strategy."""
    config_retail = expected_configuration_for_request(model_key="xgboost", domain="retail")
    assert "sample_weight_strategy" in config_retail
    assert config_retail["sample_weight_strategy"] == "spike_sensitive"


def test_retail_production_configuration_resolves_consistently() -> None:
    """Verify current Retail production configuration resolves consistently."""
    config1 = expected_configuration_for_request(model_key="xgboost", domain="retail")
    config2 = expected_configuration_for_request(
        model_key="xgboost",
        domain="retail",
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
    )

    assert config1 == config2
    assert config1["sample_weight_strategy"] == "spike_sensitive"
    assert config1["target_transform"] == "log1p"
    assert config1["max_depth"] == 3
    assert config1["learning_rate"] == 0.03


def test_identical_configurations_reuse_fingerprint() -> None:
    """Verify identical configurations produce identical fingerprints for artifact reuse."""
    config_a = expected_configuration_for_request(model_key="xgboost", domain="retail")
    config_b = expected_configuration_for_request(model_key="xgboost", domain="retail")

    assert config_fingerprint(config_a) == config_fingerprint(config_b)
