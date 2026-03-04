"""Tests for OlmoEarth and Climate Risk models."""

import torch
import pytest
from src.models.olmo_earth import (
    ModelConfig, RenewableEnergyDetector, PatchEmbedding,
    TransformerBlock, OlmoEarthBackbone, create_model, TaskType,
)


class TestModelConfig:
    def test_default_config(self):
        config = ModelConfig()
        assert config.input_channels == 12
        assert config.input_size == 224
        assert config.hidden_dim == 768
        assert config.num_layers == 12
        assert config.patch_size == 16

    def test_num_patches(self):
        config = ModelConfig(input_size=224, patch_size=16)
        assert (config.input_size // config.patch_size) ** 2 == 196


class TestPatchEmbedding:
    def test_output_shape(self):
        config = ModelConfig(hidden_dim=128, num_layers=2, num_heads=4)
        embed = PatchEmbedding(config)
        x = torch.randn(2, 12, 224, 224)
        out = embed(x)
        num_patches = (224 // 16) ** 2
        assert out.shape == (2, num_patches + 1, 128)  # +1 for CLS token

    def test_spectral_attention(self):
        config = ModelConfig(hidden_dim=128, num_layers=2, num_heads=4)
        embed = PatchEmbedding(config)
        x = torch.randn(1, 12, 224, 224)
        out = embed(x)
        assert not torch.isnan(out).any()


class TestTransformerBlock:
    def test_output_shape_preserved(self):
        config = ModelConfig(hidden_dim=128, num_heads=4)
        block = TransformerBlock(config)
        x = torch.randn(2, 197, 128)
        out = block(x)
        assert out.shape == x.shape


class TestOlmoEarthBackbone:
    def test_cls_token_output(self):
        config = ModelConfig(hidden_dim=128, num_layers=2, num_heads=4)
        backbone = OlmoEarthBackbone(config)
        x = torch.randn(2, 12, 224, 224)
        cls = backbone(x, return_all_tokens=False)
        assert cls.shape == (2, 128)

    def test_all_tokens_output(self):
        config = ModelConfig(hidden_dim=128, num_layers=2, num_heads=4)
        backbone = OlmoEarthBackbone(config)
        x = torch.randn(2, 12, 224, 224)
        cls, patches = backbone(x, return_all_tokens=True)
        assert cls.shape == (2, 128)
        assert patches.shape == (2, 196, 128)


class TestRenewableEnergyDetector:
    @pytest.fixture
    def small_config(self):
        return ModelConfig(
            hidden_dim=64, num_layers=2, num_heads=4,
            use_real_backbone=False,
        )

    def test_single_task_detection(self, small_config):
        model = RenewableEnergyDetector(small_config, task=TaskType.INSTALLATION_DETECTION)
        x = torch.randn(2, 12, 224, 224)
        out = model(x)
        assert out.shape == (2,)

    def test_single_task_classification(self, small_config):
        model = RenewableEnergyDetector(small_config, task=TaskType.ASSET_CLASSIFICATION)
        x = torch.randn(2, 12, 224, 224)
        out = model(x)
        assert out.shape == (2, 6)  # 6 energy source types

    def test_single_task_capacity(self, small_config):
        model = RenewableEnergyDetector(small_config, task=TaskType.CAPACITY_ESTIMATION)
        x = torch.randn(2, 12, 224, 224)
        out = model(x)
        assert out.shape == (2,)
        assert (out >= 0).all()  # Softplus ensures positive

    def test_multi_task(self, small_config):
        model = RenewableEnergyDetector(small_config, task="multi")
        x = torch.randn(2, 12, 224, 224)
        out = model(x)
        assert "detection" in out
        assert "classification" in out
        assert "capacity_mw" in out
        assert "segmentation" in out
        assert out["detection"].shape == (2,)
        assert out["classification"].shape == (2, 6)
        assert out["capacity_mw"].shape == (2,)

    def test_return_features(self, small_config):
        model = RenewableEnergyDetector(small_config, task="multi")
        x = torch.randn(1, 12, 224, 224)
        out = model(x, return_features=True)
        assert "features" in out

    def test_freeze_unfreeze(self, small_config):
        model = RenewableEnergyDetector(small_config, task="multi")
        model.freeze_backbone()
        for p in model.backbone.parameters():
            assert not p.requires_grad
        model.unfreeze_backbone()
        for p in model.backbone.parameters():
            assert p.requires_grad

    def test_parameter_count(self, small_config):
        model = RenewableEnergyDetector(small_config, task="multi")
        total = sum(p.numel() for p in model.parameters())
        assert total > 0


class TestCreateModel:
    def test_factory_detection(self):
        model = create_model(task="detection", model_size="nano", use_real_backbone=False)
        x = torch.randn(1, 12, 224, 224)
        out = model(x)
        assert out.shape == (1,)

    def test_factory_multi(self):
        model = create_model(task="multi", model_size="nano", use_real_backbone=False)
        x = torch.randn(1, 12, 224, 224)
        out = model(x)
        assert isinstance(out, dict)

    def test_model_sizes(self):
        for size in ["nano", "tiny", "base"]:
            model = create_model(task="detection", model_size=size, use_real_backbone=False)
            assert model is not None
