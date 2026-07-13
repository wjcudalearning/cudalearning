from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def load_library() -> ctypes.CDLL:
    dll_path = Path(__file__).resolve().parent / "cuda_vector_add.dll"
    if not dll_path.exists():
        raise FileNotFoundError(f"找不到 DLL：{dll_path}")

    return ctypes.CDLL(str(dll_path))


def configure_functions(dll: ctypes.CDLL) -> None:
    dll.cuda_get_device_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    dll.cuda_get_device_count.restype = ctypes.c_int

    dll.cuda_vector_add_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    dll.cuda_vector_add_f32.restype = ctypes.c_int

    dll.cuda_error_string.argtypes = [ctypes.c_int]
    dll.cuda_error_string.restype = ctypes.c_char_p


def error_text(dll: ctypes.CDLL, error_code: int) -> str:
    raw = dll.cuda_error_string(error_code)
    return raw.decode("utf-8", errors="replace") if raw else "Unknown CUDA error"


def main() -> int:
    try:
        dll = load_library()
        configure_functions(dll)
    except (FileNotFoundError, OSError) as exc:
        print(f"[載入失敗] {exc}")
        return 1

    device_count = ctypes.c_int()
    error = dll.cuda_get_device_count(ctypes.byref(device_count))
    if error != 0:
        print(f"[CUDA 初始化失敗] code={error}, {error_text(dll, error)}")
        print("請確認 NVIDIA 驅動已安裝，而且目前電腦有可用的 NVIDIA GPU。")
        return 2

    print(f"偵測到 CUDA GPU 數量：{device_count.value}")

    values_a = [1.0, 2.0, 3.0, 4.0]
    values_b = [10.0, 20.0, 30.0, 40.0]
    n = len(values_a)

    FloatArray = ctypes.c_float * n
    array_a = FloatArray(*values_a)
    array_b = FloatArray(*values_b)
    array_out = FloatArray()

    error = dll.cuda_vector_add_f32(array_a, array_b, array_out, n)
    if error != 0:
        print(f"[Kernel 執行失敗] code={error}, {error_text(dll, error)}")
        return 3

    result = list(array_out)
    expected = [a + b for a, b in zip(values_a, values_b)]

    print("輸入 A：", values_a)
    print("輸入 B：", values_b)
    print("GPU 結果：", result)

    if result != expected:
        print("結果不符合預期。")
        return 4

    print("測試成功：.cu → .dll → Python ctypes → CUDA kernel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
