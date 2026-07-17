from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
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
CONTRACT_FILES = {
    "manifest": Path("cuda_project.json"),
    "header": Path("include/visionflow_cuda.h"),
    "errors": Path("include/visionflow_cuda_errors.h"),
    "internal": Path("include/visionflow_cuda_internal.cuh"),
    "source": Path("visionflow_cuda.cu"),
    "smoke": Path("test_cuda_api.cu"),
    "build": Path("build_cuda_dll.ps1"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_contract(root: Path = ROOT) -> dict:
    paths = {name: root / relative for name, relative in CONTRACT_FILES.items()}
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise AssertionError(f"Missing CUDA contract files: {missing_files}")

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    texts = {
        name: path.read_text(encoding="utf-8")
        for name, path in paths.items()
        if name != "manifest"
    }

    header_exports = set(
        re.findall(r"VF_CUDA_API\s+int\s+(vf_[A-Za-z0-9_]+)\s*\(", texts["header"])
    )
    source_exports = set(
        re.findall(r"VF_CUDA_API\s+int\s+(vf_[A-Za-z0-9_]+)\s*\(", texts["source"])
    )
    abi_match = re.search(r"#define\s+VF_CUDA_ABI_VERSION\s+(\d+)", texts["header"])
    if abi_match is None:
        raise AssertionError("VF_CUDA_ABI_VERSION is missing from the public header")

    errors: list[str] = []
    if manifest.get("project_name") != "visionflow_cuda":
        errors.append("cuda_project.json project_name must be visionflow_cuda")
    if manifest.get("dll_sources") != ["visionflow_cuda.cu"]:
        errors.append("DLL source manifest must contain only visionflow_cuda.cu")
    if manifest.get("smoke_sources") != ["test_cuda_api.cu"]:
        errors.append("smoke source manifest must contain only test_cuda_api.cu")
    if header_exports != source_exports:
        errors.append(
            "header/source exports differ: "
            f"missing_definitions={sorted(header_exports - source_exports)}, "
            f"undeclared_definitions={sorted(source_exports - header_exports)}"
        )
    missing_required = REQUIRED_ABI_V1_EXPORTS - header_exports
    if missing_required:
        errors.append(f"required ABI v1 exports missing: {sorted(missing_required)}")
    missing_smoke = {name for name in REQUIRED_SMOKE_EXPORTS if name not in texts["smoke"]}
    if missing_smoke:
        errors.append(f"native smoke does not call required exports: {sorted(missing_smoke)}")
    if "visionflow_cuda.cu" not in texts["build"] or "test_cuda_api.cu" not in texts["build"]:
        errors.append("build script is missing the explicit DLL or smoke source")
    if re.search(r"(?:\*\.cu|Get-ChildItem[^\n]*\.cu)", texts["build"], re.IGNORECASE):
        errors.append("build script must not compile CUDA sources through a wildcard/glob")
    if errors:
        raise AssertionError("CUDA build preflight failed:\n- " + "\n- ".join(errors))

    return {
        "schema_version": 1,
        "project_name": manifest["project_name"],
        "abi_version": int(abi_match.group(1)),
        "architecture": manifest.get("architecture", "sm_86"),
        "exports": sorted(header_exports),
        "dll_sources": manifest["dll_sources"],
        "smoke_sources": manifest["smoke_sources"],
        "sha256": {name: _sha256(path) for name, path in paths.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Statically validate this standalone CUDA DLL project.")
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
