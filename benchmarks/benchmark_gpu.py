"""
GPU vs CPU Benchmark for OpenEnergy Engine

Measures inference throughput for:
- OlmoEarth detection model
- Climate risk model
- Batch valuation

Usage:
    python benchmarks/benchmark_gpu.py
    python benchmarks/benchmark_gpu.py --device cuda --batch-size 64
    python benchmarks/benchmark_gpu.py --device cuda --amp
"""

import argparse
import time
import torch
import numpy as np
from contextlib import contextmanager


@contextmanager
def timer(label: str, results: dict):
    """Context manager to time a block and store results."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    yield
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start
    results[label] = elapsed


def benchmark_detection(device: str, batch_size: int, use_amp: bool, warmup: int = 3, runs: int = 10):
    """Benchmark OlmoEarth detection model."""
    from src.models.olmo_earth import create_model

    print(f"\n{'='*60}")
    print(f"Detection Model Benchmark")
    print(f"  Device: {device} | Batch: {batch_size} | AMP: {use_amp}")
    print(f"{'='*60}")

    model = create_model(task="multi", model_size="base", use_real_backbone=False)
    model = model.to(device).eval()

    images = torch.randn(batch_size, 12, 224, 224, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            if use_amp and device == "cuda":
                with torch.cuda.amp.autocast():
                    _ = model(images)
            else:
                _ = model(images)

    # Benchmark
    times = []
    with torch.no_grad():
        for i in range(runs):
            results = {}
            with timer("forward", results):
                if use_amp and device == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                else:
                    outputs = model(images)
            times.append(results["forward"])

    mean_t = np.mean(times)
    std_t = np.std(times)
    throughput = batch_size / mean_t

    print(f"  Mean forward: {mean_t*1000:.1f} ms (+/- {std_t*1000:.1f})")
    print(f"  Throughput:   {throughput:.1f} images/sec")
    print(f"  Output keys:  {list(outputs.keys())}")

    if device == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak GPU mem: {mem:.0f} MB")
        torch.cuda.reset_peak_memory_stats()

    return {"mean_ms": mean_t * 1000, "std_ms": std_t * 1000, "throughput": throughput}


def benchmark_climate(device: str, batch_size: int, use_amp: bool, warmup: int = 3, runs: int = 10):
    """Benchmark climate risk model."""
    from src.models.climate_risk import create_climate_model

    print(f"\n{'='*60}")
    print(f"Climate Risk Model Benchmark")
    print(f"  Device: {device} | Batch: {batch_size} | AMP: {use_amp}")
    print(f"{'='*60}")

    model = create_climate_model()
    model = model.to(device).eval()

    location = torch.randn(batch_size, 3, device=device)
    surface_vars = torch.randn(batch_size, 7, device=device)
    scenario = torch.randint(0, 4, (batch_size,), device=device)
    target_year = torch.full((batch_size,), 2050, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(location, surface_vars, scenario, target_year)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(runs):
            results = {}
            with timer("forward", results):
                if use_amp and device == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = model(location, surface_vars, scenario, target_year)
                else:
                    outputs = model(location, surface_vars, scenario, target_year)
            times.append(results["forward"])

    mean_t = np.mean(times)
    std_t = np.std(times)
    throughput = batch_size / mean_t

    print(f"  Mean forward: {mean_t*1000:.1f} ms (+/- {std_t*1000:.1f})")
    print(f"  Throughput:   {throughput:.0f} locations/sec")

    if device == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak GPU mem: {mem:.0f} MB")
        torch.cuda.reset_peak_memory_stats()

    return {"mean_ms": mean_t * 1000, "std_ms": std_t * 1000, "throughput": throughput}


def benchmark_valuation(batch_size: int, runs: int = 10):
    """Benchmark valuation engine (CPU-bound, NumPy)."""
    from src.valuation.engine import ValuationEngine, AssetCharacteristics, AssetType

    print(f"\n{'='*60}")
    print(f"Valuation Engine Benchmark (CPU)")
    print(f"  Assets per batch: {batch_size}")
    print(f"{'='*60}")

    engine = ValuationEngine(discount_rate=0.08)

    assets = [
        AssetCharacteristics(
            asset_id=f"bench_{i}",
            asset_type=AssetType.SOLAR_UTILITY,
            latitude=35.0 + i * 0.01,
            longitude=-119.9,
            state="CA",
            capacity_mw=50 + i,
            verification_status="verified",
            verification_confidence=0.9,
        )
        for i in range(batch_size)
    ]

    # Warmup
    for a in assets[:3]:
        engine.value_asset(a)

    # Benchmark
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        for a in assets:
            engine.value_asset(a)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_t = np.mean(times)
    throughput = batch_size / mean_t

    print(f"  Mean batch:   {mean_t*1000:.1f} ms")
    print(f"  Throughput:   {throughput:.0f} valuations/sec")

    return {"mean_ms": mean_t * 1000, "throughput": throughput}


def main():
    parser = argparse.ArgumentParser(description="OpenEnergy Engine GPU Benchmark")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (AMP)")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
        args.amp = False

    print(f"PyTorch {torch.__version__}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    results = {}
    results["detection"] = benchmark_detection(args.device, args.batch_size, args.amp, runs=args.runs)
    results["climate"] = benchmark_climate(args.device, args.batch_size, args.amp, runs=args.runs)
    results["valuation"] = benchmark_valuation(args.batch_size, runs=args.runs)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"  {name:15s} | {r['mean_ms']:8.1f} ms | {r['throughput']:8.1f} /sec")


if __name__ == "__main__":
    main()
