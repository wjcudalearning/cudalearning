# visionflow_cuda

可直接放進 repository 的：

```text
cuda_projects/visionflow_cuda/
```

這個版本將正式 DLL source 與測試 source 寫在 `cuda_project.json`，GitHub Actions 不會再把所有 `.cu` 無差別編入 DLL。

## 專案內容

```text
visionflow_cuda/
├── include/
│   ├── visionflow_cuda.h
│   ├── visionflow_cuda_errors.h
│   └── visionflow_cuda_internal.cuh
├── cuda_project.json
├── visionflow_cuda.cu
├── test_cuda_api.cu
├── validate_cuda_dll.py
├── requirements-test.txt
└── build_local.ps1
```

## 本版調整

- DLL 與 `test_cuda_api.exe` 分開編譯。
- workflow 支援專案內 `include/`。
- `cuda_project.json` 明確列出 DLL 與 test sources。
- Gaussian Blur 改為水平、垂直兩段 separable kernel。
- Adaptive Mean Threshold 改為 `uint64` integral image，並處理 OpenCV 的 replicate border。
- Python 比較改為檢查「超出容許誤差的 pixel 比率」，避免單點大錯誤被放過。
- 增加多尺寸 resize、灰階/BGR Gaussian、多組 adaptive threshold、反相 threshold、morphology iterations 測試。

## GitHub Actions

將 bundle 內的：

```text
.github/workflows/build-changed-cuda-projects.yml
cuda_projects/visionflow_cuda/
```

複製到 repository 後 commit、push。

Action 會產生 artifact：

```text
visionflow_cuda-windows-x64-sm86
```

裡面包含：

```text
visionflow_cuda.dll
visionflow_cuda.lib
visionflow_cuda.exp
include/
test_cuda_api.exe
validate_cuda_dll.py
requirements-test.txt
README.md
```

GitHub-hosted Windows runner 沒有 NVIDIA GPU，因此 workflow 只編譯測試 EXE，不執行 CUDA runtime test。

## 在 RTX 3090 電腦測試

先執行 native smoke test：

```powershell
.\test_cuda_api.exe
```

再執行 OpenCV CPU/GPU 等價測試：

```powershell
python -m pip install -r .\requirements-test.txt
python .\validate_cuda_dll.py --dll .\visionflow_cuda.dll --json .\validation.json
```

也可在 x64 Native Tools PowerShell 直接從 source 編譯：

```powershell
.\build_local.ps1 -Architecture sm_86 -RunNativeTest -RunPythonTest
```
