# VisionFlow CUDA DLL standalone project

This directory is designed for the repository workflow under `cuda_projects/`.

Canonical layout:

```text
visionflow_cuda/
├── include/
│   ├── visionflow_cuda.h
│   ├── visionflow_cuda_errors.h
│   └── visionflow_cuda_internal.cuh
├── build_cuda_dll.ps1
├── cuda_project.json
├── preflight_cuda_build.py
├── test_cuda_api.cu
├── validate_cuda_dll.py
└── visionflow_cuda.cu
```

The build script explicitly compiles only `visionflow_cuda.cu` into the DLL and
compiles `test_cuda_api.cu` into a separate executable. The `.cuh` file is included
by the DLL source; it is not a separate compilation unit.

The static preflight has no OpenCV dependency. It accepts the canonical `include/`
layout and also a root-level flattened header layout. Conflicting duplicate copies
are rejected so a stale header cannot be used accidentally.

GitHub-hosted Windows runners compile and inspect the DLL but do not execute the GPU
smoke test because they do not provide an NVIDIA GPU.
