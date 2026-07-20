# VisionFlow CUDA DLL

## Preprocessing architecture

Detectors must describe preprocessing with the backend-neutral operators in `core/preprocess_plan.py`. The CPU executor defines OpenCV fallback semantics; the CUDA executor may use the generic native plan, reusable primitives, or a compatible fused adapter. New detectors should compose existing operators instead of adding detector-named DLL exports.

The optional versioned `VfPlanDescV1` ABI supports detector-neutral linear Gray, non-expanding single-channel Resize(area), Gaussian, Threshold, AdaptiveMean and Morphology graphs. Resize expansion or mixed-axis scaling remains unsupported because it does not share OpenCV's decimation semantics. `vf_plan_query/create/execute/destroy` validate and compile a complete graph before execution. A supported plan uses a context-owned non-blocking CUDA stream, persistent scratch and morphology ping-pong buffers, one asynchronous host-to-device upload, continuous kernels, and one necessary asynchronous device-to-host download followed by a single stream synchronization. An unsupported operator rejects the whole plan before any CUDA primitive executes.

The additive `VfDagPlanDescV1` ABI uses the same detector-neutral operators with topological input-node references and an explicit output-node list. `vf_dag_plan_query/create/execute/destroy` upload the root once, keep branch intermediates in context-owned grow-only device buffers, copy only requested outputs, and synchronize once. Detector 900 uses this route to share device grayscale across its outer and inner masks.

When exported, `vf_context_last_timings` returns the most recent persistent-context operation in `VfCudaTimingsV1`. CUDA events separate H2D or resident D2D staging, kernels, D2H, Gaussian, Adaptive Mean, threshold and morphology time; host clocks report context creation, allocation, synchronization and ROI-batch release overhead. The Python runtime exposes these values as `performance_stats()["native_timings_ms"]`.

`vf_preprocess_401_2_u8` remains an additive ABI v1 compatibility adapter. It is not the template for future detector APIs.

`visionflow_cuda.dll` 是 AOI 的可選 CUDA backend。Recipe 未勾選 GPU 時不載入它；勾選但 DLL/裝置不可用時，依 `fallback_to_cpu` 決定回退 CPU 或明確失敗。

目前 Gaussian blur 使用 horizontal/vertical separable kernels 與 constant weights；Adaptive Mean Threshold 使用 replicate-border 64-bit integral image。公開 C ABI 維持 v1，因此 Python bridge 與既有打包版介面不需修改，但更新原始碼後必須重新編譯 DLL。

新版 DLL 另外提供可選的 persistent context 與 generic plan exports。context 持有 non-blocking stream、grow-only scratch 與 morphology ping-pong buffers；支援的 linear plan 會把中間影像保留在 GPU。Batch/monitor 透過 `GpuExecutionSession` 跨影像重用同一個 runtime/context。Detector 401-2 的 fused export 繼續作為舊 DLL compatibility adapter；Python bridge 若載入舊 DLL，會自動保留 fused、stateless primitive 或 CPU fallback 路徑。

## 檔案

```text
gpu/
├── include/
│   ├── visionflow_cuda.h            # 公開、穩定的 C ABI
│   ├── visionflow_cuda_errors.h     # 錯誤碼
│   └── visionflow_cuda_internal.cuh # .cu 內部 CUDA helper
├── visionflow_cuda.cu               # 正式 DLL kernels 與 exports
├── test_cuda_api.cu                 # C++ ABI/device smoke
├── preflight_cuda_build.py          # ABI/source/runtime/build 靜態契約與 hash manifest
├── validate_cuda_dll.py             # OpenCV 與 AOI CPU/GPU 比對
└── build_cuda_dll.ps1               # Windows 編譯與測試入口
```

## RTX 3090 編譯

安裝 NVIDIA Driver、CUDA Toolkit、Visual Studio 2022 C++ Build Tools 後，在 x64 Native Tools PowerShell 執行：

```powershell
.\gpu\build_cuda_dll.ps1 -Architecture sm_86
```

輸出：

- `gpu/visionflow_cuda.dll`
- `gpu/visionflow_cuda.lib`
- `gpu/test_cuda_api.exe`

build script 明確使用 static CUDA runtime；`nvcuda.dll` 仍由 NVIDIA Driver 提供。

編譯前會先執行 `preflight_cuda_build.py`，核對 public header、`.cu` definitions、Python bridge、native smoke 與明確 source manifest，並將 SHA-256 manifest 寫到 `outputs_validation/cuda_build_preflight.json`。DLL、LIB 與 smoke EXE 會先在 ignored staging 目錄完成，通過 `dumpbin /exports` 與 `/dependents` 後才取代正式 artifacts，避免失敗時誤用半成品或 stale DLL。

## 編譯並測試

測試 ABI、structured primitive matrix，以及 4K grayscale、Gaussian k45、Adaptive Mean b35 的 CPU/GPU benchmark：

```powershell
.\gpu\build_cuda_dll.ps1 -RunTests
```

再加一張真實影像與 recipe，會同時執行 CPU/GPU AOI 結果比對：

```powershell
.\gpu\build_cuda_dll.ps1 -RunTests `
  -Image C:\AOI_TEST\sample.png `
  -Recipe .\recipes\PRODUCT_A_AOI_01.yaml
```

完整 CPU/GPU roadmap、CUDA 環境、五個 recipe、GUI、打包與壓力測試清單統一見 [`Todo.md`](../Todo.md)。
