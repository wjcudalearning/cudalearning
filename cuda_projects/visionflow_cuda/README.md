# visionflow_cuda — workflow-ready CUDA DLL project

這個資料夾已整理成可直接放進既有多專案 CUDA workflow 的形式。

## 放置位置

將整個資料夾放到 repository：

```text
cuda_projects/
└── visionflow_cuda/
    ├── visionflow_cuda.cu
    ├── visionflow_cuda.h
    ├── visionflow_cuda_errors.h
    ├── visionflow_cuda_internal.cuh
    ├── test.py
    ├── preflight_cuda_build.py
    ├── build_cuda_dll.ps1
    ├── tests/
    │   └── test_cuda_api.cu
    └── integration_tools/
```

**資料夾不可改名**。你的 workflow 使用第一層資料夾名稱作為 DLL 名稱，因此輸出會是：

```text
visionflow_cuda.dll
```

## 為什麼這個版本可直接丟進 workflow

- 專案根目錄只有一個 `.cu`：`visionflow_cuda.cu`，不會把含 `main()` 的測試程式誤編進 DLL。
- 所有 DLL 編譯需要的標頭都放在專案根目錄，支援只使用 `-I cuda_projects/visionflow_cuda` 的 workflow。
- `tests/test_cuda_api.cu` 與 DLL source 分離，可由支援測試 EXE 的新版 workflow 額外編譯。
- `test.py` 只使用 Python 標準函式庫，不依賴 OpenCV、NumPy 或原 AOI 專案的 `core/`。
- `preflight_cuda_build.py` 是專案內自包含檢查，不再依賴 repository 其他模組。
- 公開 header 同時接受 `VISIONFLOW_CUDA_EXPORTS` 與 workflow 常用的 `CUDA_DLL_EXPORTS`。

## GitHub Actions 行為

一般 changed-project workflow 只需要執行類似：

```bat
nvcc -std=c++17 -O2 --shared --cudart static -arch=sm_86 ^
  -Xcompiler "/MD /EHsc" ^
  -DCUDA_DLL_EXPORTS ^
  -I"cuda_projects\visionflow_cuda" ^
  "cuda_projects\visionflow_cuda\visionflow_cuda.cu" ^
  -o "build\visionflow_cuda\visionflow_cuda.dll"
```

GitHub-hosted Windows runner 通常沒有 NVIDIA GPU，因此 workflow 應負責**編譯與匯出檢查**，不要把實際 GPU 執行測試當成必要關卡。

## 本機完整建置

在已安裝 CUDA Toolkit 與 Visual Studio C++ Build Tools 的 x64 Native Tools PowerShell：

```powershell
.\build_cuda_dll.ps1 -Architecture sm_86 -BuildSmoke
```

有 NVIDIA GPU 時執行 DLL smoke test：

```powershell
.\build_cuda_dll.ps1 -Architecture sm_86 -RunTests
```

無 GPU、只想確認 DLL 能載入時：

```powershell
.\build_cuda_dll.ps1 -Architecture sm_86 -RunTests -AllowNoGpu
```

也可直接測 artifact：

```powershell
python .\test.py --dll .\visionflow_cuda.dll
```

## integration_tools

`integration_tools/validate_cuda_dll_full.py`、benchmark gate 與 production manifest 是原 AOI repository 的完整整合驗證工具，會依賴 `core/`、OpenCV、NumPy、PyYAML、recipe 與實際影像；它們不參與一般 CUDA DLL workflow 編譯。
