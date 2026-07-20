from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ABI_V1_EXPORTS = {
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
REQUIRED_SMOKE_EXPORTS = {
    "vf_bgr_to_gray_u8",
    "vf_context_create",
    "vf_context_destroy",
    "vf_context_stats",
    "vf_gpu_abi_version",
    "vf_gpu_compute_capability",
    "vf_gpu_device_count",
    "vf_gpu_device_name",
    "vf_gpu_error_message",
    "vf_preprocess_401_2_u8",
}
OPTIONAL_GENERIC_PLAN_EXPORTS = {
    "vf_plan_query",
    "vf_plan_create",
    "vf_plan_execute",
    "vf_plan_destroy",
    "vf_dag_plan_query",
    "vf_dag_plan_create",
    "vf_dag_plan_execute",
    "vf_dag_plan_destroy",
}
OPTIONAL_RESIDENT_ROI_EXPORTS = {
    "vf_context_upload_u8",
    "vf_plan_execute_roi",
    "vf_dag_plan_execute_roi",
}
OPTIONAL_ROI_BATCH_EXPORTS = {
    "vf_gpu_memory_info",
    "vf_roi_batch_create",
    "vf_roi_batch_info",
    "vf_roi_batch_download_u8",
    "vf_roi_batch_destroy",
}
OPTIONAL_TIMING_EXPORTS = {"vf_context_last_timings"}
CONTRACT_FILES = {
    "header": Path("gpu/include/visionflow_cuda.h"),
    "source": Path("gpu/visionflow_cuda.cu"),
    "smoke": Path("gpu/test_cuda_api.cu"),
    "build": Path("gpu/build_cuda_dll.ps1"),
    "preflight": Path("gpu/preflight_cuda_build.py"),
    "runtime": Path("core/gpu_runtime.py"),
    "validator": Path("gpu/validate_cuda_dll.py"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_contract(root: Path = ROOT) -> dict:
    paths = {name: root / relative for name, relative in CONTRACT_FILES.items()}
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise AssertionError(f"Missing CUDA contract files: {missing_files}")

    texts = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    header_exports = set(
        re.findall(r"VF_CUDA_API\s+int\s+(vf_[A-Za-z0-9_]+)\s*\(", texts["header"])
    )
    source_exports = set(
        re.findall(r"VF_CUDA_API\s+int\s+(vf_[A-Za-z0-9_]+)\s*\(", texts["source"])
    )
    abi_match = re.search(r"#define\s+VF_CUDA_ABI_VERSION\s+(\d+)", texts["header"])
    if abi_match is None:
        raise AssertionError("VF_CUDA_ABI_VERSION is missing from the public header")

    errors = []
    if header_exports != source_exports:
        errors.append(
            "header/source exports differ: "
            f"missing_definitions={sorted(header_exports - source_exports)}, "
            f"undeclared_definitions={sorted(source_exports - header_exports)}"
        )
    missing_required = REQUIRED_ABI_V1_EXPORTS - header_exports
    if missing_required:
        errors.append(f"required ABI v1 exports missing: {sorted(missing_required)}")
    missing_smoke = {
        name for name in REQUIRED_SMOKE_EXPORTS if name not in texts["smoke"]
    }
    if missing_smoke:
        errors.append(f"native smoke does not call required exports: {sorted(missing_smoke)}")
    missing_runtime = {name for name in REQUIRED_ABI_V1_EXPORTS if name not in texts["runtime"]}
    if missing_runtime:
        errors.append(f"Python runtime does not reference exports: {sorted(missing_runtime)}")
    declared_plan_exports = OPTIONAL_GENERIC_PLAN_EXPORTS & header_exports
    if declared_plan_exports and declared_plan_exports != OPTIONAL_GENERIC_PLAN_EXPORTS:
        errors.append("generic plan ABI exports must be declared as one complete optional set")
    if declared_plan_exports:
        missing_plan_smoke = {name for name in OPTIONAL_GENERIC_PLAN_EXPORTS if name not in texts["smoke"]}
        missing_plan_runtime = {name for name in OPTIONAL_GENERIC_PLAN_EXPORTS if name not in texts["runtime"]}
        if missing_plan_smoke:
            errors.append(f"native smoke does not call generic plan exports: {sorted(missing_plan_smoke)}")
        if missing_plan_runtime:
            errors.append(f"Python runtime does not reference generic plan exports: {sorted(missing_plan_runtime)}")
    declared_resident_exports = OPTIONAL_RESIDENT_ROI_EXPORTS & header_exports
    if declared_resident_exports and declared_resident_exports != OPTIONAL_RESIDENT_ROI_EXPORTS:
        errors.append("resident image/ROI exports must be declared as one complete optional set")
    if declared_resident_exports:
        missing_resident_smoke = {name for name in OPTIONAL_RESIDENT_ROI_EXPORTS if name not in texts["smoke"]}
        missing_resident_runtime = {name for name in OPTIONAL_RESIDENT_ROI_EXPORTS if name not in texts["runtime"]}
        if missing_resident_smoke:
            errors.append(f"native smoke does not call resident ROI exports: {sorted(missing_resident_smoke)}")
        if missing_resident_runtime:
            errors.append(f"Python runtime does not reference resident ROI exports: {sorted(missing_resident_runtime)}")
    declared_batch_exports = OPTIONAL_ROI_BATCH_EXPORTS & header_exports
    if declared_batch_exports and declared_batch_exports != OPTIONAL_ROI_BATCH_EXPORTS:
        errors.append("ROI batch exports must be declared as one complete optional set")
    if declared_batch_exports:
        missing_batch_smoke = {name for name in OPTIONAL_ROI_BATCH_EXPORTS if name not in texts["smoke"]}
        missing_batch_runtime = {name for name in OPTIONAL_ROI_BATCH_EXPORTS if name not in texts["runtime"]}
        if missing_batch_smoke:
            errors.append(f"native smoke does not call ROI batch exports: {sorted(missing_batch_smoke)}")
        if missing_batch_runtime:
            errors.append(f"Python runtime does not reference ROI batch exports: {sorted(missing_batch_runtime)}")
    declared_timing_exports = OPTIONAL_TIMING_EXPORTS & header_exports
    if declared_timing_exports:
        missing_timing_smoke = {name for name in OPTIONAL_TIMING_EXPORTS if name not in texts["smoke"]}
        missing_timing_runtime = {name for name in OPTIONAL_TIMING_EXPORTS if name not in texts["runtime"]}
        if missing_timing_smoke:
            errors.append(f"native smoke does not call timing exports: {sorted(missing_timing_smoke)}")
        if missing_timing_runtime:
            errors.append(f"Python runtime does not reference timing exports: {sorted(missing_timing_runtime)}")
    if "visionflow_cuda.cu" not in texts["build"] or "test_cuda_api.cu" not in texts["build"]:
        errors.append("build script is missing an explicit DLL or smoke source manifest")
    if re.search(r"(?:\*\.cu|Get-ChildItem[^\n]*\.cu)", texts["build"], re.IGNORECASE):
        errors.append("build script must not compile CUDA sources through a wildcard/glob")
    if errors:
        raise AssertionError("CUDA build preflight failed:\n- " + "\n- ".join(errors))

    return {
        "schema_version": 1,
        "abi_version": int(abi_match.group(1)),
        "exports": sorted(header_exports),
        "optional_generic_plan_exports": sorted(declared_plan_exports),
        "optional_resident_roi_exports": sorted(declared_resident_exports),
        "optional_roi_batch_exports": sorted(declared_batch_exports),
        "optional_timing_exports": sorted(declared_timing_exports),
        "dll_sources": ["gpu/visionflow_cuda.cu"],
        "smoke_sources": ["gpu/test_cuda_api.cu"],
        "sha256": {
            name: _sha256(path) for name, path in paths.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Statically validate the CUDA build contract.")
    parser.add_argument("--output", type=Path, help="Optional JSON manifest output path.")
    args = parser.parse_args()
    result = inspect_contract()
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
