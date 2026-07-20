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

# Headers may be kept in include/ (preferred) or directly in the project root.
# Supporting both layouts makes the project robust when files were uploaded through
# the GitHub web UI and the directory hierarchy was accidentally flattened.
FILE_CANDIDATES = {
    "header": (Path("include/visionflow_cuda.h"), Path("visionflow_cuda.h")),
    "errors": (Path("include/visionflow_cuda_errors.h"), Path("visionflow_cuda_errors.h")),
    "internal": (
        Path("include/visionflow_cuda_internal.cuh"),
        Path("visionflow_cuda_internal.cuh"),
    ),
    "source": (Path("visionflow_cuda.cu"),),
    "smoke": (Path("test_cuda_api.cu"),),
    "build": (Path("build_cuda_dll.ps1"),),
}

DEFAULT_MANIFEST = {
    "schema_version": 1,
    "project_name": "visionflow_cuda",
    "architecture": "sm_86",
    "dll_sources": ["visionflow_cuda.cu"],
    "smoke_sources": ["test_cuda_api.cu"],
    "output_dll": "visionflow_cuda.dll",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _nearby_names(root: Path, expected_name: str) -> list[str]:
    stem = Path(expected_name).stem.split(".")[0]
    candidates = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and (stem in path.stem or expected_name.lower() == path.name.lower())
    }
    return sorted(candidates)


def _resolve_file(root: Path, label: str, candidates: tuple[Path, ...]) -> Path:
    existing = [root / relative for relative in candidates if (root / relative).is_file()]
    if not existing:
        expected = " or ".join(relative.as_posix() for relative in candidates)
        nearby = _nearby_names(root, candidates[-1].name)
        hint = f" Similar files found: {nearby}." if nearby else ""
        raise AssertionError(f"Missing {label}: expected {expected}.{hint}")

    # A stale root-level copy can silently shadow include/. Fail early when duplicate
    # files differ; identical duplicates are accepted but include/ remains preferred.
    if len(existing) > 1:
        hashes = {_sha256(path) for path in existing}
        if len(hashes) != 1:
            names = [_relative(path, root) for path in existing]
            raise AssertionError(
                f"Conflicting duplicate {label} files: {names}. "
                "Delete the stale copy or make the files identical."
            )

    return existing[0]


def _load_manifest(root: Path) -> tuple[dict, Path | None]:
    manifest_path = root / "cuda_project.json"
    if not manifest_path.is_file():
        return dict(DEFAULT_MANIFEST), None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssertionError(f"Invalid cuda_project.json: {exc}") from exc
    return manifest, manifest_path


def inspect_contract(root: Path = ROOT) -> dict:
    root = root.resolve()
    paths = {
        label: _resolve_file(root, label, candidates)
        for label, candidates in FILE_CANDIDATES.items()
    }
    manifest, manifest_path = _load_manifest(root)

    texts = {
        name: path.read_text(encoding="utf-8")
        for name, path in paths.items()
    }

    header_exports = set(
        re.findall(r"VF_CUDA_API\s+int\s+(vf_[A-Za-z0-9_]+)\s*\(", texts["header"])
    )
    source_exports = set(
        re.findall(r"VF_CUDA_API\s+int\s+(vf_[A-Za-z0-9_]+)\s*\(", texts["source"])
    )
    abi_match = re.search(r"#define\s+VF_CUDA_ABI_VERSION\s+(\d+)", texts["header"])
    if abi_match is None:
        raise AssertionError("VF_CUDA_ABI_VERSION is missing from visionflow_cuda.h")

    errors: list[str] = []

    expected_manifest = DEFAULT_MANIFEST
    for key in ("schema_version", "project_name", "dll_sources", "smoke_sources", "output_dll"):
        if manifest.get(key) != expected_manifest[key]:
            errors.append(
                f"cuda_project.json {key} must be {expected_manifest[key]!r}; "
                f"got {manifest.get(key)!r}"
            )

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

    required_includes = {
        "source": ("visionflow_cuda.h", "visionflow_cuda_internal.cuh"),
        "header": ("visionflow_cuda_errors.h",),
        "internal": ("visionflow_cuda_errors.h", "cuda_runtime.h"),
        "smoke": ("visionflow_cuda.h",),
    }
    for text_name, include_names in required_includes.items():
        for include_name in include_names:
            pattern = rf'#include\s*[<"]{re.escape(include_name)}[>"]'
            if not re.search(pattern, texts[text_name]):
                errors.append(f"{paths[text_name].name} must include {include_name}")

    if "visionflow_cuda.cu" not in texts["build"] or "test_cuda_api.cu" not in texts["build"]:
        errors.append("build script must explicitly name the DLL and smoke sources")
    if re.search(r"(?:\*\.cu|Get-ChildItem[^\n]*\.cu)", texts["build"], re.IGNORECASE):
        errors.append("build script must not compile CUDA sources through a wildcard/glob")

    if errors:
        raise AssertionError("CUDA build preflight failed:\n- " + "\n- ".join(errors))

    include_dirs: list[str] = []
    for key in ("header", "errors", "internal"):
        parent = paths[key].parent
        relative_parent = _relative(parent, root) if parent != root else "."
        if relative_parent not in include_dirs:
            include_dirs.append(relative_parent)
    if "." not in include_dirs:
        include_dirs.append(".")

    hashed_paths = dict(paths)
    if manifest_path is not None:
        hashed_paths["manifest"] = manifest_path

    return {
        "schema_version": 1,
        "project_name": "visionflow_cuda",
        "abi_version": int(abi_match.group(1)),
        "architecture": manifest.get("architecture", "sm_86"),
        "exports": sorted(header_exports),
        "dll_sources": ["visionflow_cuda.cu"],
        "smoke_sources": ["test_cuda_api.cu"],
        "output_dll": "visionflow_cuda.dll",
        "include_dirs": include_dirs,
        "resolved_files": {
            name: _relative(path, root) for name, path in paths.items()
        },
        "manifest_source": (
            _relative(manifest_path, root) if manifest_path is not None else "built-in defaults"
        ),
        "sha256": {
            name: _sha256(path) for name, path in sorted(hashed_paths.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Statically validate the standalone VisionFlow CUDA DLL project."
    )
    parser.add_argument("--output", type=Path, help="Optional JSON manifest output path.")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Project root. Defaults to the directory containing this script.",
    )
    args = parser.parse_args()

    result = inspect_contract(args.root)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
