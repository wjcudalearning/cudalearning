# visionflow_cuda — GitHub Actions 專案版

此資料夾可直接放在 repository 的：

```text
cuda_projects/visionflow_cuda/
```

資料夾名稱、`cuda_project.json` 的 `output_name` 與輸出的 DLL 名稱必須一致：`visionflow_cuda`。

## Action 編譯內容

- DLL：只編譯 `visionflow_cuda.cu`
- 測試 EXE：只編譯 `test_cuda_api.cu`，並連結 `visionflow_cuda.lib`
- include：使用 `include/`
- 產物：`visionflow_cuda.dll`、`visionflow_cuda.lib`、`visionflow_cuda.exp`、`test_cuda_api.exe`
- 額外檢查：header/source export 一致性、DLL exports、DLL dependencies

`test_cuda_api.exe` 在 GitHub hosted runner 上只編譯、不執行，因為 `windows-2022` runner 沒有 NVIDIA GPU。下載 artifact 後，請在 RTX 3090 或其他 CUDA GPU 機器執行。

## 為什麼不再把所有 `.cu` 丟進 DLL

`cuda_project.json` 明確分開：

```json
"dll_sources": ["visionflow_cuda.cu"],
"test_targets": [
  {
    "name": "test_cuda_api",
    "sources": ["test_cuda_api.cu"]
  }
]
```

因此 `test_cuda_api.cu` 不會被錯誤連進 DLL，也支援未來在 `dll_sources` 明確加入多個真正的 DLL CUDA source。

## 本機編譯

在已載入 Visual Studio x64 Native Tools、且 `nvcc` 可用的 PowerShell 執行：

```powershell
.\cuda_projects\visionflow_cuda\build_cuda_dll.ps1
```

預設輸出到：

```text
build/visionflow_cuda/
```

## Python 驗證工具

`validate_cuda_dll.py`、`profile_401_pipeline.py` 等工具仍保留，但依賴完整 AOI repository 的 `core/`、OpenCV、NumPy 與 PyYAML；GitHub hosted runner 不會執行 GPU benchmark。DLL 編譯與 C ABI 測試 EXE 編譯不依賴 OpenCV。
