from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_NAME = PROJECT_ROOT.name
EXPECTED_PROJECT_NAME = "visionflow_cuda"

REQUIRED_EXPORTS = {
    "vf_adaptive_mean_u8",
    "vf_bgr_to_gray_u8",
    "vf_bgr_to_rgb_u8",
    "vf_context_create",
    "vf_context_destroy",
    "vf_context_stats",
    "vf_crop_u8",
    "vf_gaussian_blur_u8",
    "vf_gpu_abi_version",
    "vf_gpu_compute_capability",
    "vf_gpu_device_count",
    "vf_gpu_device_name",
    "vf_gpu_error_message",
    "vf_morphology_rect_u8",
    "vf_preprocess_401_2_u8",
    "vf_resize_gray_u8",
    "vf_threshold_u8",
}

CONTRACT_FILES = {
    "header": Path("visionflow_cuda.h"),
    "errors": Path("visionflow_cuda_errors.h"),
    "internal": Path("visionflow_cuda_internal.cuh"),
    "source": Path("visionflow_cuda.cu"),
    "smoke": Path("tests/test_cuda_api.cu"),
    "python_test": Path("test.py"),
    "build": Path("build_cuda_dll.ps1"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_contract() -> dict:
    errors: list[str] = []

    if PROJECT_NAME != EXPECTED_PROJECT_NAME:
        errors.append(
            f"project folder must be named '{EXPECTED_PROJECT_NAME}' so the workflow "
            f"produces {EXPECTED_PROJECT_NAME}.dll; current folder is '{PROJECT_NAME}'"
        )

    paths = {name: PROJECT_ROOT / relative for name, relative in CONTRACT_FILES.items()}
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in paths.values() if not path.is_file()]
    if missing:
        errors.append(f"missing required project files: {missing}")

    root_cuda_sources = sorted(path.name for path in PROJECT_ROOT.glob("*.cu"))
    if root_cuda_sources != ["visionflow_cuda.cu"]:
        errors.append(
            "the project root must contain only visionflow_cuda.cu; put executable/smoke .cu "
            f"files under tests/. Found: {root_cuda_sources}"
        )

    if errors:
        raise AssertionError("CUDA workflow preflight failed:\n- " + "\n- ".join(errors))

    texts = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    header_exports = set(
        re.findall(r"VF_CUDA_API\s+int\s+(vf_[A-Za-z0-9_]+)\s*\(", texts["header"])
    )
    source_exports = set(
        re.findall(r"VF_CUDA_API\s+int\s+(vf_[A-Za-z0-9_]+)\s*\(", texts["source"])
    )

    abi_match = re.search(r"#define\s+VF_CUDA_ABI_VERSION\s+(\d+)", texts["header"])
    if abi_match is None:
        errors.append("VF_CUDA_ABI_VERSION is missing from visionflow_cuda.h")

    if header_exports != source_exports:
        errors.append(
            "header/source exports differ: "
            f"missing_definitions={sorted(header_exports - source_exports)}, "
            f"undeclared_definitions={sorted(source_exports - header_exports)}"
        )

    missing_required = REQUIRED_EXPORTS - header_exports
    if missing_required:
        errors.append(f"required ABI exports missing: {sorted(missing_required)}")

    smoke_calls = {name for name in header_exports if name in texts["smoke"]}
    if not REQUIRED_EXPORTS.intersection(smoke_calls):
        errors.append("native smoke source does not reference the public CUDA API")

    if "visionflow_cuda.cu" not in texts["build"] or "tests/test_cuda_api.cu" not in texts["build"].replace("\\", "/"):
        errors.append("build_cuda_dll.ps1 must explicitly list DLL and smoke sources")

    if re.search(r"(?:\*\.cu|Get-ChildItem[^\n]*\.cu)", texts["build"], re.IGNORECASE):
        errors.append("build_cuda_dll.ps1 must not compile CUDA sources using a wildcard")

    for expected in ("vf_gpu_abi_version", "vf_gpu_device_count", "vf_threshold_u8"):
        if expected not in texts["python_test"]:
            errors.append(f"test.py does not reference required smoke API: {expected}")

    if errors:
        raise AssertionError("CUDA workflow preflight failed:\n- " + "\n- ".join(errors))

    return {
        "schema_version": 1,
        "project": PROJECT_NAME,
        "dll_name": f"{PROJECT_NAME}.dll",
        "abi_version": int(abi_match.group(1)),
        "dll_sources": ["visionflow_cuda.cu"],
        "smoke_sources": ["tests/test_cuda_api.cu"],
        "exports": sorted(header_exports),
        "sha256": {name: sha256(path) for name, path in paths.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate this CUDA project before GitHub Actions build.")
    parser.add_argument("--output", type=Path, help="Optional JSON manifest output path.")
    args = parser.parse_args()

    result = inspect_contract()
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
