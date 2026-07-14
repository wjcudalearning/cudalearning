from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

VF_CUDA_OK = 0
VF_MORPH_OPEN = 0
VF_MORPH_CLOSE = 1
VF_MORPH_DILATE = 2
VF_MORPH_ERODE = 3

U8_PTR = ctypes.POINTER(ctypes.c_uint8)


@dataclass(frozen=True)
class Comparison:
    name: str
    max_diff: int
    mean_diff: float
    excess_ratio: float


class VisionFlowCuda:
    def __init__(self, dll_path: Path) -> None:
        self.dll_path = dll_path.resolve()
        loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
        self.lib = loader(str(self.dll_path))
        self._bind()

        if self.lib.vf_gpu_abi_version() != 1:
            raise RuntimeError(
                f"Unsupported ABI version: {self.lib.vf_gpu_abi_version()}"
            )
        if self.lib.vf_gpu_device_count() <= 0:
            raise RuntimeError("No CUDA device is available")

    def _bind(self) -> None:
        self.lib.vf_gpu_abi_version.argtypes = []
        self.lib.vf_gpu_abi_version.restype = ctypes.c_int
        self.lib.vf_gpu_device_count.argtypes = []
        self.lib.vf_gpu_device_count.restype = ctypes.c_int
        self.lib.vf_gpu_compute_capability.argtypes = []
        self.lib.vf_gpu_compute_capability.restype = ctypes.c_int
        self.lib.vf_gpu_device_name.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.vf_gpu_device_name.restype = ctypes.c_int
        self.lib.vf_gpu_error_message.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self.lib.vf_gpu_error_message.restype = ctypes.c_int

        image_args = [
            U8_PTR,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            U8_PTR,
            ctypes.c_int,
            ctypes.c_int,
        ]
        for name in ("vf_bgr_to_gray_u8", "vf_bgr_to_rgb_u8"):
            function = getattr(self.lib, name)
            function.argtypes = image_args
            function.restype = ctypes.c_int

        self.lib.vf_crop_u8.argtypes = image_args + [ctypes.c_int] * 4
        self.lib.vf_crop_u8.restype = ctypes.c_int
        self.lib.vf_resize_gray_u8.argtypes = image_args + [ctypes.c_int] * 2
        self.lib.vf_resize_gray_u8.restype = ctypes.c_int
        self.lib.vf_gaussian_blur_u8.argtypes = image_args + [ctypes.c_int]
        self.lib.vf_gaussian_blur_u8.restype = ctypes.c_int
        self.lib.vf_threshold_u8.argtypes = image_args + [ctypes.c_int] * 3
        self.lib.vf_threshold_u8.restype = ctypes.c_int
        self.lib.vf_adaptive_mean_u8.argtypes = image_args + [
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.vf_adaptive_mean_u8.restype = ctypes.c_int
        self.lib.vf_morphology_rect_u8.argtypes = image_args + [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.vf_morphology_rect_u8.restype = ctypes.c_int

    @property
    def device_name(self) -> str:
        output = ctypes.create_string_buffer(256)
        self._check(self.lib.vf_gpu_device_name(output, len(output)))
        return output.value.decode("utf-8", errors="replace")

    @property
    def compute_capability(self) -> int:
        return int(self.lib.vf_gpu_compute_capability())

    def _check(self, result: int) -> None:
        if result == VF_CUDA_OK:
            return
        output = ctypes.create_string_buffer(512)
        self.lib.vf_gpu_error_message(result, output, len(output))
        message = output.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"CUDA DLL call failed: {message} (code={result})")

    @staticmethod
    def _input(image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            raise TypeError(f"Expected uint8, got {image.dtype}")
        return np.ascontiguousarray(image)

    @staticmethod
    def _pointer(image: np.ndarray) -> U8_PTR:
        return image.ctypes.data_as(U8_PTR)

    @staticmethod
    def _channels(image: np.ndarray) -> int:
        return 1 if image.ndim == 2 else int(image.shape[2])

    def _call_image(
        self,
        function: Callable[..., int],
        src: np.ndarray,
        dst: np.ndarray,
        *extra: object,
    ) -> np.ndarray:
        src = self._input(src)
        if not dst.flags.c_contiguous:
            raise ValueError("Output must be contiguous")
        height, width = src.shape[:2]
        result = function(
            self._pointer(src),
            width,
            height,
            int(src.strides[0]),
            self._channels(src),
            self._pointer(dst),
            int(dst.strides[0]),
            self._channels(dst),
            *extra,
        )
        self._check(result)
        return dst

    def bgr_to_gray(self, image: np.ndarray) -> np.ndarray:
        image = self._input(image)
        dst = np.empty(image.shape[:2], dtype=np.uint8)
        return self._call_image(self.lib.vf_bgr_to_gray_u8, image, dst)

    def bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        image = self._input(image)
        dst = np.empty_like(image)
        return self._call_image(self.lib.vf_bgr_to_rgb_u8, image, dst)

    def crop(
        self, image: np.ndarray, x: int, y: int, width: int, height: int
    ) -> np.ndarray:
        image = self._input(image)
        shape = (height, width) if image.ndim == 2 else (height, width, image.shape[2])
        dst = np.empty(shape, dtype=np.uint8)
        return self._call_image(
            self.lib.vf_crop_u8, image, dst, x, y, width, height
        )

    def resize_gray(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        image = self._input(image)
        dst = np.empty((height, width), dtype=np.uint8)
        return self._call_image(
            self.lib.vf_resize_gray_u8, image, dst, width, height
        )

    def gaussian_blur(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        image = self._input(image)
        dst = np.empty_like(image)
        return self._call_image(
            self.lib.vf_gaussian_blur_u8, image, dst, kernel_size
        )

    def threshold(
        self,
        image: np.ndarray,
        threshold: int,
        max_value: int = 255,
        invert: bool = False,
    ) -> np.ndarray:
        image = self._input(image)
        dst = np.empty_like(image)
        return self._call_image(
            self.lib.vf_threshold_u8,
            image,
            dst,
            threshold,
            max_value,
            int(invert),
        )

    def adaptive_threshold(
        self,
        image: np.ndarray,
        block_size: int,
        c: float,
        max_value: int = 255,
        invert: bool = False,
    ) -> np.ndarray:
        image = self._input(image)
        dst = np.empty_like(image)
        return self._call_image(
            self.lib.vf_adaptive_mean_u8,
            image,
            dst,
            block_size,
            ctypes.c_float(c),
            max_value,
            int(invert),
        )

    def morphology(
        self,
        image: np.ndarray,
        operation: int,
        kernel_size: int,
        iterations: int,
    ) -> np.ndarray:
        image = self._input(image)
        dst = np.empty_like(image)
        return self._call_image(
            self.lib.vf_morphology_rect_u8,
            image,
            dst,
            operation,
            kernel_size,
            iterations,
        )


def compare(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    max_diff: int = 0,
    max_excess_ratio: float = 0.0,
) -> Comparison:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"{name}: shape/dtype mismatch "
            f"actual={actual.shape}/{actual.dtype}, "
            f"expected={expected.shape}/{expected.dtype}"
        )

    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    observed_max = int(delta.max(initial=0))
    excess_ratio = float(np.count_nonzero(delta > max_diff) / max(delta.size, 1))
    result = Comparison(
        name=name,
        max_diff=observed_max,
        mean_diff=round(float(delta.mean()), 6),
        excess_ratio=round(excess_ratio, 8),
    )
    if excess_ratio > max_excess_ratio:
        raise AssertionError(
            f"{name}: observed_max={observed_max}, "
            f"pixels_over_tolerance={excess_ratio:.8f}, "
            f"allowed={max_excess_ratio:.8f}, tolerance={max_diff}"
        )
    print(f"PASS {name}: {result}")
    return result


def validate_primitives(runtime: VisionFlowCuda) -> list[Comparison]:
    rng = np.random.default_rng(20260714)
    bgr = rng.integers(0, 256, size=(137, 211, 3), dtype=np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    metrics: list[Comparison] = []

    metrics.append(
        compare(
            "bgr_to_rgb",
            runtime.bgr_to_rgb(bgr),
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        )
    )
    metrics.append(
        compare(
            "bgr_to_gray",
            runtime.bgr_to_gray(bgr),
            gray,
            max_diff=1,
            max_excess_ratio=0.0,
        )
    )
    metrics.append(
        compare(
            "crop_bgr",
            runtime.crop(bgr, 17, 13, 91, 67),
            bgr[13:80, 17:108],
        )
    )
    metrics.append(
        compare(
            "crop_gray",
            runtime.crop(gray, 9, 11, 73, 59),
            gray[11:70, 9:82],
        )
    )

    for dst_width, dst_height in ((96, 64), (137, 91), (71, 119)):
        metrics.append(
            compare(
                f"resize_area_{dst_width}x{dst_height}",
                runtime.resize_gray(gray, dst_width, dst_height),
                cv2.resize(
                    gray,
                    (dst_width, dst_height),
                    interpolation=cv2.INTER_AREA,
                ),
                max_diff=1,
                max_excess_ratio=0.002,
            )
        )

    for kernel_size in (3, 5, 9):
        metrics.append(
            compare(
                f"gaussian_gray_k{kernel_size}",
                runtime.gaussian_blur(gray, kernel_size),
                cv2.GaussianBlur(
                    gray,
                    (kernel_size, kernel_size),
                    0,
                    borderType=cv2.BORDER_REFLECT_101,
                ),
                max_diff=2,
                max_excess_ratio=0.0005,
            )
        )
        metrics.append(
            compare(
                f"gaussian_bgr_k{kernel_size}",
                runtime.gaussian_blur(bgr, kernel_size),
                cv2.GaussianBlur(
                    bgr,
                    (kernel_size, kernel_size),
                    0,
                    borderType=cv2.BORDER_REFLECT_101,
                ),
                max_diff=2,
                max_excess_ratio=0.0005,
            )
        )

    for invert in (False, True):
        threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        expected = cv2.threshold(gray, 128, 255, threshold_type)[1]
        metrics.append(
            compare(
                f"global_threshold_invert_{int(invert)}",
                runtime.threshold(gray, 128, 255, invert),
                expected,
            )
        )

    for block_size, c_value, invert in (
        (3, 2.0, False),
        (11, 2.0, False),
        (35, -2.0, False),
        (11, 2.0, True),
        (35, -2.0, True),
    ):
        threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        expected = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            threshold_type,
            block_size,
            c_value,
        )
        metrics.append(
            compare(
                f"adaptive_mean_b{block_size}_c{c_value}_inv{int(invert)}",
                runtime.adaptive_threshold(
                    gray, block_size, c_value, 255, invert
                ),
                expected,
                max_diff=0,
                max_excess_ratio=0.0,
            )
        )

    binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    operations = (
        ("open", VF_MORPH_OPEN, cv2.MORPH_OPEN),
        ("close", VF_MORPH_CLOSE, cv2.MORPH_CLOSE),
        ("dilate", VF_MORPH_DILATE, cv2.MORPH_DILATE),
        ("erode", VF_MORPH_ERODE, cv2.MORPH_ERODE),
    )
    for name, vf_operation, cv_operation in operations:
        for iterations in (1, 2):
            if cv_operation in (cv2.MORPH_OPEN, cv2.MORPH_CLOSE):
                expected = cv2.morphologyEx(
                    binary, cv_operation, kernel, iterations=iterations
                )
            elif cv_operation == cv2.MORPH_DILATE:
                expected = cv2.dilate(binary, kernel, iterations=iterations)
            else:
                expected = cv2.erode(binary, kernel, iterations=iterations)
            metrics.append(
                compare(
                    f"morphology_{name}_iter{iterations}",
                    runtime.morphology(
                        binary, vf_operation, 3, iterations
                    ),
                    expected,
                )
            )

    return metrics


def benchmark(runtime: VisionFlowCuda, repetitions: int) -> dict[str, object]:
    if repetitions <= 0:
        return {}
    image = np.random.default_rng(7).integers(
        0, 256, size=(2160, 3840, 3), dtype=np.uint8
    )
    runtime.bgr_to_gray(image)
    started = time.perf_counter()
    for _ in range(repetitions):
        runtime.bgr_to_gray(image)
    elapsed = time.perf_counter() - started
    result: dict[str, object] = {
        "operation": "bgr_to_gray_4k_including_transfer",
        "repetitions": repetitions,
        "total_sec": round(elapsed, 4),
        "average_ms": round(elapsed * 1000.0 / repetitions, 3),
    }
    print(f"BENCHMARK {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare visionflow_cuda.dll primitives with OpenCV CPU output."
    )
    parser.add_argument(
        "--dll",
        type=Path,
        default=Path("visionflow_cuda.dll"),
        help="Path to visionflow_cuda.dll.",
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=10,
        help="4K grayscale benchmark repetitions; use 0 to disable.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path for a JSON result report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = VisionFlowCuda(args.dll)
    capability = runtime.compute_capability
    print(
        f"CUDA DLL ready: device={runtime.device_name}, "
        f"capability={capability // 10}.{capability % 10}, "
        f"path={runtime.dll_path}"
    )
    comparisons = validate_primitives(runtime)
    benchmark_result = benchmark(runtime, args.benchmark)
    report = {
        "dll": str(runtime.dll_path),
        "device": runtime.device_name,
        "compute_capability": capability,
        "comparisons": [item.__dict__ for item in comparisons],
        "benchmark": benchmark_result,
    }
    if args.json:
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote report: {args.json}")
    print("All CUDA/OpenCV equivalence validations passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
