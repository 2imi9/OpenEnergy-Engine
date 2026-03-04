"""Tests for valuation engine."""

import pytest
from src.valuation.engine import (
    ValuationEngine, AssetCharacteristics, AssetType,
    ValuationResult, TokenizationMetrics, ClimateRiskFactors,
)


class TestValuationEngine:
    @pytest.fixture
    def engine(self):
        return ValuationEngine(discount_rate=0.08)

    @pytest.fixture
    def solar_asset(self):
        return AssetCharacteristics(
            asset_id="test_solar_001",
            asset_type=AssetType.SOLAR_UTILITY,
            latitude=35.0,
            longitude=-119.9,
            state="CA",
            capacity_mw=100,
            verification_status="verified",
            verification_confidence=0.92,
        )

    @pytest.fixture
    def wind_asset(self):
        return AssetCharacteristics(
            asset_id="test_wind_001",
            asset_type=AssetType.WIND_ONSHORE,
            latitude=32.0,
            longitude=-100.0,
            state="TX",
            capacity_mw=200,
            verification_status="pending",
            verification_confidence=0.5,
        )

    def test_value_asset_returns_result(self, engine, solar_asset):
        result = engine.value_asset(solar_asset)
        assert isinstance(result, ValuationResult)
        assert result.asset_id == "test_solar_001"

    def test_npv_is_finite(self, engine, solar_asset):
        result = engine.value_asset(solar_asset)
        assert result.npv_usd != float("inf")
        assert result.npv_usd != float("-inf")

    def test_irr_reasonable(self, engine, solar_asset):
        result = engine.value_asset(solar_asset)
        assert -1.0 < result.irr < 1.0  # IRR between -100% and 100%

    def test_lcoe_positive(self, engine, solar_asset):
        result = engine.value_asset(solar_asset)
        assert result.lcoe_per_mwh > 0

    def test_annual_arrays_length(self, engine, solar_asset):
        result = engine.value_asset(solar_asset)
        assert len(result.annual_generation_mwh) == 25
        assert len(result.annual_revenue_usd) == 25
        assert len(result.annual_costs_usd) == 25
        assert len(result.annual_net_cash_flow) == 25

    def test_generation_degrades(self, engine, solar_asset):
        result = engine.value_asset(solar_asset)
        assert result.annual_generation_mwh[0] > result.annual_generation_mwh[-1]

    def test_verified_higher_npv_than_pending(self, engine):
        verified = AssetCharacteristics(
            asset_id="v", asset_type=AssetType.SOLAR_UTILITY,
            latitude=35.0, longitude=-119.9, state="CA", capacity_mw=100,
            verification_status="verified", verification_confidence=0.95,
        )
        pending = AssetCharacteristics(
            asset_id="p", asset_type=AssetType.SOLAR_UTILITY,
            latitude=35.0, longitude=-119.9, state="CA", capacity_mw=100,
            verification_status="pending", verification_confidence=0.5,
        )
        v_result = engine.value_asset(verified)
        p_result = engine.value_asset(pending)
        assert v_result.verification_adjusted_npv > p_result.verification_adjusted_npv

    def test_climate_risk_reduces_npv(self, engine, solar_asset):
        no_risk = engine.value_asset(solar_asset)
        with_risk = engine.value_asset(
            solar_asset, climate_risk=ClimateRiskFactors(risk_score=0.8)
        )
        assert with_risk.risk_adjusted_npv < no_risk.risk_adjusted_npv

    def test_wind_asset(self, engine, wind_asset):
        result = engine.value_asset(wind_asset)
        assert isinstance(result, ValuationResult)
        assert result.npv_usd != 0


class TestTokenization:
    def test_tokenization_metrics(self):
        engine = ValuationEngine()
        asset = AssetCharacteristics(
            asset_id="tok_test", asset_type=AssetType.SOLAR_UTILITY,
            latitude=35.0, longitude=-119.9, state="CA", capacity_mw=50,
            verification_status="verified", verification_confidence=0.9,
        )
        valuation = engine.value_asset(asset)
        tokens = engine.calculate_tokenization_metrics(valuation, total_tokens=1_000_000)
        assert isinstance(tokens, TokenizationMetrics)
        assert tokens.total_tokens == 1_000_000
        assert tokens.value_per_token_usd > 0
        assert tokens.risk_rating in ("AAA", "AA", "A", "BBB", "BB", "B")


class TestCalculateNPV:
    def test_positive_cash_flows(self):
        engine = ValuationEngine(discount_rate=0.10)
        npv = engine.calculate_npv([100, 100, 100], initial_investment=200)
        assert npv > 0

    def test_zero_discount_rate(self):
        engine = ValuationEngine(discount_rate=0.0)
        npv = engine.calculate_npv([100, 100], initial_investment=200)
        assert abs(npv) < 1e-6  # NPV ~= 0

    def test_negative_npv(self):
        engine = ValuationEngine(discount_rate=0.50)
        npv = engine.calculate_npv([10, 10, 10], initial_investment=100)
        assert npv < 0


class TestCalculateIRR:
    def test_known_irr(self):
        engine = ValuationEngine()
        # Investment of 100, return of 110 after 1 year => IRR = 10%
        irr = engine.calculate_irr([110], initial_investment=100)
        assert abs(irr - 0.10) < 0.01

    def test_irr_positive(self):
        engine = ValuationEngine()
        irr = engine.calculate_irr([50, 50, 50], initial_investment=100)
        assert irr > 0
