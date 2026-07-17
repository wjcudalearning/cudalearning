from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpu_runtime import GpuRuntime  # noqa: E402
from core.pipeline import AOIPipeline  # noqa: E402
from core.recipe_manager import RecipeManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate visionflow_cuda.dll against the CPU AOI path.")
    parser.add_argument("--dll", default="gpu/visionflow_cuda.dll", help="CUDA DLL path.")
    parser.add_argument("--image", help="Optional real image for full CPU/GPU pipeline comparison.")
    parser.add_argument("--recipe", help="Recipe used with --image.")
    parser.add_argument("--benchmark", type=int, default=20, help="Primitive benchmark repetitions.")
    args = parser.parse_args()
    if bool(args.image) != bool(args.recipe):
        parser.error("--image and --recipe must be provided together")
    return args


def compare(name: str, actual: np.ndarray, expected: np.ndarray, max_diff: int = 0, mismatch_ratio: float = 0.0) -> dict:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"{name}: shape/dtype mismatch actual={actual.shape}/{actual.dtype}, expected={expected.shape}/{expected.dtype}"
        )
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    observed_max = int(delta.max(initial=0))
    observed_ratio = float(np.count_nonzero(delta) / max(delta.size, 1))
    out_of_tolerance_ratio = float(np.count_nonzero(delta > max_diff) / max(delta.size, 1))
    if out_of_tolerance_ratio > mismatch_ratio:
        raise AssertionError(
            f"{name}: max_diff={observed_max} (limit {max_diff}), mismatch_ratio={observed_ratio:.6f} "
            f"out_of_tolerance_ratio={out_of_tolerance_ratio:.6f} (limit {mismatch_ratio:.6f})"
        )
    result = {
        "name": name,
        "max_diff": observed_max,
        "mean_diff": round(float(delta.mean()), 6),
        "mismatch_ratio": round(observed_ratio, 6),
        "out_of_tolerance_ratio": round(out_of_tolerance_ratio, 6),
    }
    print(f"PASS {name}: {result}")
    return result


def validate_primitives(runtime: GpuRuntime) -> list[dict]:
    rng = np.random.default_rng(20260714)
    bgr = rng.integers(0, 256, size=(128, 192, 3), dtype=np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
    metrics = []
    metrics.append(compare("bgr_to_rgb", runtime.bgr_to_rgb(bgr), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    metrics.append(compare("bgr_to_gray", runtime.bgr_to_gray(bgr), gray, max_diff=1))
    metrics.append(compare("crop_bgr", runtime.crop(bgr, 17, 13, 91, 67), bgr[13:80, 17:108]))
    metrics.append(
        compare(
            "resize_gray",
            runtime.resize_gray(gray, 96, 64),
            cv2.resize(gray, (96, 64), interpolation=cv2.INTER_AREA),
            max_diff=1,
            mismatch_ratio=0.001,
        )
    )
    metrics.append(
        compare(
            "gaussian_blur_gray",
            runtime.gaussian_blur(gray, 5),
            cv2.GaussianBlur(gray, (5, 5), 0),
            max_diff=2,
            mismatch_ratio=0.001,
        )
    )
    metrics.append(compare("global_threshold", runtime.threshold(gray, 128, 255, False), binary))
    structured_gray = {
        "random_odd": rng.integers(0, 256, size=(65, 97), dtype=np.uint8),
        "black": np.zeros((63, 79), dtype=np.uint8),
        "white": np.full((63, 79), 255, dtype=np.uint8),
        "checker": ((np.indices((63, 79)).sum(axis=0) % 2) * 255).astype(np.uint8),
        "non_contiguous": gray[:, ::2],
    }
    for case_name, case in structured_gray.items():
        for kernel_size in (3, 5, 15, 25, 45):
            expected_gaussian = cv2.GaussianBlur(case, (kernel_size, kernel_size), 0)
            metrics.append(
                compare(
                    f"gaussian_{case_name}_k{kernel_size}",
                    runtime.gaussian_blur(case, kernel_size),
                    expected_gaussian,
                    max_diff=2,
                    mismatch_ratio=0.001,
                )
            )
    expected_gaussian_bgr = cv2.GaussianBlur(bgr, (15, 15), 0)
    metrics.append(
        compare(
            "gaussian_bgr_k15",
            runtime.gaussian_blur(bgr, 15),
            expected_gaussian_bgr,
            max_diff=2,
            mismatch_ratio=0.001,
        )
    )
    adaptive_cases = (
        ("random_binary_b3_c2", structured_gray["random_odd"], 3, 2.0, False),
        ("random_binary_b11_cneg2", structured_gray["random_odd"], 11, -2.0, False),
        ("random_inverse_b35_c24", structured_gray["random_odd"], 35, 2.4, True),
        ("black_binary_b11", structured_gray["black"], 11, 2.0, False),
        ("white_inverse_b11", structured_gray["white"], 11, 2.0, True),
        ("checker_binary_b35", structured_gray["checker"], 35, -2.0, False),
        ("non_contiguous_inverse_b11", structured_gray["non_contiguous"], 11, -2.0, True),
    )
    for case_name, case, block_size, adaptive_c, invert in adaptive_cases:
        threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        expected_adaptive = cv2.adaptiveThreshold(
            case,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            threshold_type,
            block_size,
            adaptive_c,
        )
        metrics.append(
            compare(
                f"adaptive_{case_name}",
                runtime.adaptive_threshold(case, block_size, adaptive_c, 255, invert),
                expected_adaptive,
                max_diff=0,
                mismatch_ratio=0.02,
            )
        )
    if runtime.supports_fused_401_2:
        fused_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        fused_expected = cv2.adaptiveThreshold(
            cv2.GaussianBlur(fused_gray, (25, 25), 0),
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            35,
            -2.0,
        )
        metrics.append(
            compare(
                "fused_401_2_bgr",
                runtime.preprocess_401_2(bgr, 25, 35, -2.0, 255, True),
                fused_expected,
                max_diff=0,
                mismatch_ratio=0.02,
            )
        )
        first_context_stats = runtime.performance_stats()["persistent_context"]
        repeated = runtime.preprocess_401_2(bgr, 25, 35, -2.0, 255, True)
        metrics.append(
            compare(
                "fused_401_2_bgr_reused_context",
                repeated,
                fused_expected,
                max_diff=0,
                mismatch_ratio=0.02,
            )
        )
        second_context_stats = runtime.performance_stats()["persistent_context"]
        if first_context_stats.get("allocation_count") != second_context_stats.get("allocation_count"):
            raise AssertionError(
                "fused_401_2 context allocated again for an unchanged image shape: "
                f"first={first_context_stats}, second={second_context_stats}"
            )
        print(f"PASS fused_401_2 persistent context reuse: {second_context_stats}")
    else:
        print(f"SKIP fused_401_2_bgr: {runtime.fused_unavailable_reason}")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for operation, cv_operation in (
        ("open", cv2.MORPH_OPEN),
        ("close", cv2.MORPH_CLOSE),
        ("dilate", cv2.MORPH_DILATE),
        ("erode", cv2.MORPH_ERODE),
    ):
        expected = (
            cv2.morphologyEx(binary, cv_operation, kernel, iterations=1)
            if operation in {"open", "close"}
            else cv2.dilate(binary, kernel, iterations=1)
            if operation == "dilate"
            else cv2.erode(binary, kernel, iterations=1)
        )
        metrics.append(compare(f"morphology_{operation}", runtime.morphology(binary, operation, 3, 1), expected))
    return metrics


def _average_ms(operation, repetitions: int) -> float:
    started = time.perf_counter()
    for _ in range(repetitions):
        operation()
    return (time.perf_counter() - started) * 1000.0 / repetitions


def benchmark(runtime: GpuRuntime, repetitions: int) -> dict:
    if repetitions <= 0:
        return {}
    image = np.random.default_rng(7).integers(0, 256, size=(2160, 3840, 3), dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    runtime.bgr_to_gray(image)
    runtime.gaussian_blur(gray, 45)
    runtime.adaptive_threshold(gray, 35, -2.0, 255, False)
    operations = (
        ("bgr_to_gray_4k", lambda: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), lambda: runtime.bgr_to_gray(image)),
        ("gaussian_gray_4k_k45", lambda: cv2.GaussianBlur(gray, (45, 45), 0), lambda: runtime.gaussian_blur(gray, 45)),
        (
            "adaptive_mean_gray_4k_b35",
            lambda: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, -2.0),
            lambda: runtime.adaptive_threshold(gray, 35, -2.0, 255, False),
        ),
    )
    if runtime.supports_fused_401_2:
        operations += (
            (
                "fused_401_2_bgr_4k",
                lambda: cv2.adaptiveThreshold(
                    cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (25, 25), 0),
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY_INV,
                    35,
                    -2.0,
                ),
                lambda: runtime.preprocess_401_2(image, 25, 35, -2.0, 255, True),
            ),
        )
    measurements = []
    for name, cpu_operation, gpu_operation in operations:
        cpu_ms = _average_ms(cpu_operation, repetitions)
        gpu_ms = _average_ms(gpu_operation, repetitions)
        measurements.append(
            {
                "operation": name,
                "cpu_average_ms": round(cpu_ms, 3),
                "gpu_average_ms_including_transfer": round(gpu_ms, 3),
                "speedup": round(cpu_ms / gpu_ms, 3) if gpu_ms > 0 else None,
            }
        )
    result = {
        "repetitions": repetitions,
        "image_shape": list(image.shape),
        "measurements": measurements,
        "gpu_host_metrics": runtime.performance_stats(),
    }
    print(f"BENCHMARK {result}")
    return result


def normalized_result(result: dict) -> dict:
    normalized = deepcopy(result)
    for key in ("duration_sec", "outputs", "execution"):
        normalized.pop(key, None)
    for tile_result in normalized.get("tiles", []):
        for detector_result in tile_result.get("detectors", []):
            detector_result.pop("execution", None)
    return normalized


def validate_pipeline(image_path: Path, recipe_path: Path, dll_path: str) -> None:
    manager = RecipeManager()
    base = manager.load(recipe_path)
    cpu_recipe = deepcopy(base)
    gpu_recipe = deepcopy(base)
    cpu_recipe["gpu"] = {
        "tiling": False,
        "display": False,
        "dll_path": dll_path,
        "fallback_to_cpu": False,
    }
    gpu_recipe["gpu"] = {
        "tiling": True,
        "display": True,
        "dll_path": dll_path,
        "fallback_to_cpu": False,
    }
    for config in cpu_recipe.get("detectors", {}).values():
        config["use_gpu"] = False
    for config in gpu_recipe.get("detectors", {}).values():
        config["use_gpu"] = bool(config.get("enabled", False))
    for recipe in (cpu_recipe, gpu_recipe):
        recipe["output"] = {key: False for key in recipe.get("output", {})}

    with tempfile.TemporaryDirectory(prefix="visionflow_cuda_validation_") as temporary:
        temporary_path = Path(temporary)
        cpu_path = temporary_path / "cpu.yaml"
        gpu_path = temporary_path / "gpu.yaml"
        cpu_path.write_text(yaml.safe_dump(cpu_recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
        gpu_path.write_text(yaml.safe_dump(gpu_recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
        cpu_result = AOIPipeline(cpu_path, temporary_path / "cpu_outputs").run(image_path)
        gpu_result = AOIPipeline(gpu_path, temporary_path / "gpu_outputs").run(image_path)

    active = gpu_result.get("execution", {}).get("gpu", {})
    if not active.get("tiling", {}).get("active"):
        raise AssertionError(f"GPU tiling did not activate: {active}")
    inactive_detectors = {
        detector_id: status
        for detector_id, status in active.get("detectors", {}).items()
        if status.get("requested") and not status.get("active")
    }
    if inactive_detectors:
        raise AssertionError(f"GPU detectors did not activate: {inactive_detectors}")

    cpu_normalized = normalized_result(cpu_result)
    gpu_normalized = normalized_result(gpu_result)
    if cpu_normalized != gpu_normalized:
        summary = {
            "cpu_final": cpu_result.get("final_result"),
            "gpu_final": gpu_result.get("final_result"),
            "cpu_summary": cpu_result.get("summary"),
            "gpu_summary": gpu_result.get("summary"),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise AssertionError("Full pipeline CPU/GPU results differ; inspect the printed summaries and report JSON")
    print("PASS full_pipeline: CPU and GPU inspection results are identical")


def main() -> int:
    args = parse_args()
    runtime = GpuRuntime(args.dll, fallback_to_cpu=False)
    if not runtime.available:
        raise SystemExit(f"CUDA DLL unavailable: {runtime.unavailable_reason}")
    print(
        f"CUDA DLL ready: device={runtime.device_name}, capability={runtime.compute_capability}, "
        f"path={runtime.dll_path}"
    )
    validate_primitives(runtime)
    benchmark(runtime, args.benchmark)
    if args.image and args.recipe:
        validate_pipeline(Path(args.image), Path(args.recipe), str(runtime.dll_path))
    print("All requested CUDA validations passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
