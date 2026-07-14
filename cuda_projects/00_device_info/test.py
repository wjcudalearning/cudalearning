from __future__ import annotations

import ctypes
import sys
from pathlib import Path


class CudaDeviceInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("compute_major", ctypes.c_int),
        ("compute_minor", ctypes.c_int),
        ("multiprocessor_count", ctypes.c_int),
        ("max_threads_per_block", ctypes.c_int),
        ("warp_size", ctypes.c_int),
        ("total_global_memory_bytes", ctypes.c_ulonglong),
    ]


def error_text(dll: ctypes.CDLL, code: int) -> str:
    raw = dll.cuda_error_string(code)
    return raw.decode("utf-8", errors="replace") if raw else "Unknown CUDA error"


def main() -> int:
    dll_path = Path(__file__).resolve().parent / "00_device_info.dll"
    if not dll_path.exists():
        print(f"找不到 DLL：{dll_path}")
        return 1

    try:
        dll = ctypes.CDLL(str(dll_path))
    except OSError as exc:
        print(f"DLL 載入失敗：{exc}")
        return 2

    dll.cuda_get_device_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    dll.cuda_get_device_count.restype = ctypes.c_int
    dll.cuda_get_device_info.argtypes = [ctypes.c_int, ctypes.POINTER(CudaDeviceInfo)]
    dll.cuda_get_device_info.restype = ctypes.c_int
    dll.cuda_error_string.argtypes = [ctypes.c_int]
    dll.cuda_error_string.restype = ctypes.c_char_p

    count = ctypes.c_int()
    code = dll.cuda_get_device_count(ctypes.byref(count))
    if code != 0:
        print(f"CUDA 錯誤 {code}：{error_text(dll, code)}")
        return 3
    if count.value <= 0:
        print("找不到可用的 CUDA GPU。")
        return 4

    print(f"CUDA 裝置數量：{count.value}")
    for device_index in range(count.value):
        info = CudaDeviceInfo()
        code = dll.cuda_get_device_info(device_index, ctypes.byref(info))
        if code != 0:
            print(f"讀取裝置 {device_index} 失敗：{error_text(dll, code)}")
            return 5

        name = bytes(info.name).split(b"\0", 1)[0].decode("utf-8", errors="replace")
        gib = info.total_global_memory_bytes / (1024**3)
        print(
            f"[{device_index}] {name}\n"
            f"  Compute capability: {info.compute_major}.{info.compute_minor}\n"
            f"  VRAM: {gib:.2f} GiB\n"
            f"  SM count: {info.multiprocessor_count}\n"
            f"  Max threads/block: {info.max_threads_per_block}\n"
            f"  Warp size: {info.warp_size}"
        )

    print("測試成功。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
