"""Tests for climate risk model."""

import torch
import pytest
from src.models.climate_risk import (
    ClimateConfig, ClimateRiskModel, ClimateRiskOutput,
    ClimateScenario, SolarResourceEncoder, WindResourceEncoder,
    ExtremeEventPredictor, ClimateProjector, create_climate_model,
)


class TestClimateConfig:
    def test_defaults(self):
        config = ClimateConfig()
        assert config.grid_resolution_km == 25.0
        assert config.forecast_years == 30
        assert len(config.surface_variables) == 7
        assert len(config.pressure_levels) == 9


class TestSolarResourceEncoder:
    def test_output_shape(self):
        config = ClimateConfig(hidden_dim=64)
        encoder = SolarResourceEncoder(config)
        location = torch.randn(4, 3)
        climate_vars = torch.randn(4, 7)
        month = torch.randint(0, 12, (4,))
        out = encoder(location, climate_vars, month)
        assert out.shape == (4, 64)


class TestWindResourceEncoder:
    def test_output_shape(self):
        config = ClimateConfig(hidden_dim=64)
        encoder = WindResourceEncoder(config)
        location = torch.randn(4, 3)
        climate_vars = torch.randn(4, 7)
        roughness = torch.randn(4, 1)
        hub_height = torch.ones(4, 1) * 100
        out = encoder(location, climate_vars, roughness, hub_height)
        assert out.shape == (4, 64)


class TestExtremeEventPredictor:
    def test_output_keys(self):
        config = ClimateConfig(hidden_dim=64)
        predictor = ExtremeEventPredictor(config)
        x = torch.randn(2, 64)
        probs = predictor(x)
        assert "wildfire" in probs
        assert "flood" in probs
        assert "drought" in probs
        for v in probs.values():
            assert (v >= 0).all() and (v <= 1).all()


class TestClimateProjector:
    def test_output_keys(self):
        config = ClimateConfig(hidden_dim=64)
        projector = ClimateProjector(config)
        baseline = torch.randn(2, 64)
        scenario = torch.tensor([0, 1])
        year = torch.tensor([2040, 2050])
        out = projector(baseline, scenario, year)
        assert "temperature_delta_c" in out
        assert "precipitation_delta_pct" in out
        assert "solar_delta_pct" in out
        assert "wind_delta_pct" in out


class TestClimateRiskModel:
    @pytest.fixture
    def model(self):
        return ClimateRiskModel(ClimateConfig(hidden_dim=64, num_layers=2))

    def test_forward(self, model):
        B = 4
        location = torch.randn(B, 3)
        surface_vars = torch.randn(B, 7)
        scenario = torch.randint(0, 4, (B,))
        target_year = torch.full((B,), 2050)
        out = model(location, surface_vars, scenario, target_year)
        assert "risk_score" in out
        assert out["risk_score"].shape == (B,)
        assert (out["risk_score"] >= 0).all() and (out["risk_score"] <= 1).all()

    def test_assess_risk(self, model):
        risk = model.assess_risk(
            latitude=35.0, longitude=-119.9, elevation=500,
            asset_type="solar", scenario=ClimateScenario.SSP245,
            target_year=2050,
        )
        assert isinstance(risk, ClimateRiskOutput)
        assert 0 <= risk.risk_score <= 1
        assert risk.latitude == 35.0
        assert "p50" in risk.solar_ghi_kwh_m2_year
        assert "p50" in risk.wind_speed_m_s
        assert len(risk.extreme_event_probs) == 7

    def test_all_scenarios(self, model):
        for scenario in ClimateScenario:
            risk = model.assess_risk(
                latitude=40.0, longitude=-105.0,
                scenario=scenario, target_year=2040,
            )
            assert 0 <= risk.risk_score <= 1


class TestCreateClimateModel:
    def test_factory(self):
        model = create_climate_model()
        assert isinstance(model, ClimateRiskModel)

    def test_custom_config(self):
        config = ClimateConfig(hidden_dim=32, num_layers=1)
        model = create_climate_model(config)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0
