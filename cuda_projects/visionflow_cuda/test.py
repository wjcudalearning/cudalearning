from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path


PROJECT_NAME = "visionflow_cuda"
VF_CUDA_OK = 0
VF_CUDA_ABI_VERSION = 1


def candidate_dlls(explicit: str | None) -> list[Path]:
    project_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("CUDA_DLL_PATH"):
        candidates.append(Path(os.environ["CUDA_DLL_PATH"]))
    candidates.extend(
        [
            project_dir / f"{PROJECT_NAME}.dll",
            project_dir / "build" / f"{PROJECT_NAME}.dll",
            project_dir.parents[1] / "build" / PROJECT_NAME / f"{PROJECT_NAME}.dll",
            Path.cwd() / f"{PROJECT_NAME}.dll",
            Path.cwd() / "build" / PROJECT_NAME / f"{PROJECT_NAME}.dll",
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.expanduser().resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(Path(resolved))
    return unique


def find_dll(explicit: str | None) -> Path:
    candidates = candidate_dlls(explicit)
    for path in candidates:
        if path.is_file():
            return path
    rendered = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"{PROJECT_NAME}.dll not found. Checked:\n{rendered}")


def configure(dll: ctypes.CDLL) -> None:
    u8p = ctypes.POINTER(ctypes.c_uint8)

    dll.vf_gpu_abi_version.argtypes = []
    dll.vf_gpu_abi_version.restype = ctypes.c_int
    dll.vf_gpu_device_count.argtypes = []
    dll.vf_gpu_device_count.restype = ctypes.c_int
    dll.vf_gpu_compute_capability.argtypes = []
    dll.vf_gpu_compute_capability.restype = ctypes.c_int
    dll.vf_gpu_device_name.argtypes = [ctypes.c_char_p, ctypes.c_int]
    dll.vf_gpu_device_name.restype = ctypes.c_int
    dll.vf_gpu_error_message.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    dll.vf_gpu_error_message.restype = ctypes.c_int
    dll.vf_threshold_u8.argtypes = [
        u8p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        u8p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    dll.vf_threshold_u8.restype = ctypes.c_int


def error_message(dll: ctypes.CDLL, code: int) -> str:
    buffer = ctypes.create_string_buffer(512)
    dll.vf_gpu_error_message(code, buffer, len(buffer))
    return buffer.value.decode("utf-8", errors="replace")


def run_threshold_test(dll: ctypes.CDLL) -> None:
    width, height = 8, 4
    values = [0, 31, 63, 95, 127, 128, 200, 255] * height
    source_type = ctypes.c_uint8 * len(values)
    output_type = ctypes.c_uint8 * len(values)
    source = source_type(*values)
    output = output_type()

    result = dll.vf_threshold_u8(
        source,
        width,
        height,
        width,
        1,
        output,
        width,
        1,
        127,
        255,
        0,
    )
    if result != VF_CUDA_OK:
        raise RuntimeError(f"vf_threshold_u8 failed: {error_message(dll, result)} ({result})")

    expected = [255 if value > 127 else 0 for value in values]
    actual = list(output)
    if actual != expected:
        raise AssertionError(f"threshold mismatch\nexpected={expected}\nactual={actual}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone ctypes smoke test for visionflow_cuda.dll")
    parser.add_argument("--dll", help="Path to visionflow_cuda.dll")
    parser.add_argument(
        "--allow-no-gpu",
        action="store_true",
        help="Return success when the DLL loads but the machine has no CUDA device.",
    )
    args = parser.parse_args()

    if os.name != "nt":
        print("SKIP: this artifact is a Windows DLL and test.py must run on Windows.")
        return 0

    dll_path = find_dll(args.dll)
    dll = ctypes.WinDLL(str(dll_path))
    configure(dll)

    abi = dll.vf_gpu_abi_version()
    if abi != VF_CUDA_ABI_VERSION:
        raise AssertionError(f"ABI mismatch: DLL={abi}, expected={VF_CUDA_ABI_VERSION}")

    count = dll.vf_gpu_device_count()
    print(f"DLL: {dll_path}")
    print(f"ABI: {abi}")
    print(f"CUDA device count: {count}")
    if count <= 0:
        message = "DLL loaded successfully, but no CUDA device is available."
        if args.allow_no_gpu:
            print(f"SKIP: {message}")
            return 0
        raise RuntimeError(message)

    name = ctypes.create_string_buffer(256)
    result = dll.vf_gpu_device_name(name, len(name))
    if result != VF_CUDA_OK:
        raise RuntimeError(f"vf_gpu_device_name failed: {error_message(dll, result)} ({result})")
    capability = dll.vf_gpu_compute_capability()
    print(f"Device: {name.value.decode('utf-8', errors='replace')}")
    print(f"Compute capability: {capability // 10}.{capability % 10}")

    run_threshold_test(dll)
    print("PASS: DLL load, ABI, device query, and threshold CPU/GPU equivalence smoke test")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
