from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def main() -> int:
    dll_path = Path(__file__).resolve().parent / "02_threshold.dll"
    if not dll_path.exists():
        print(f"找不到 DLL：{dll_path}")
        return 1

    try:
        dll = ctypes.CDLL(str(dll_path))
    except OSError as exc:
        print(f"DLL 載入失敗：{exc}")
        return 2

    u8_p = ctypes.POINTER(ctypes.c_ubyte)
    dll.cuda_threshold_u8.argtypes = [
        u8_p, u8_p, ctypes.c_int, ctypes.c_int, ctypes.c_ubyte, ctypes.c_int
    ]
    dll.cuda_threshold_u8.restype = ctypes.c_int
    dll.cuda_error_string.argtypes = [ctypes.c_int]
    dll.cuda_error_string.restype = ctypes.c_char_p

    values = [0, 20, 80, 127, 128, 129, 180, 255]
    expected = [0, 0, 0, 0, 0, 255, 255, 255]
    Array = ctypes.c_ubyte * len(values)
    source = Array(*values)
    output = Array()

    code = dll.cuda_threshold_u8(source, output, len(values), 1, 128, 0)
    if code != 0:
        raw = dll.cuda_error_string(code)
        message = raw.decode("utf-8", errors="replace") if raw else "Unknown CUDA error"
        print(f"CUDA 錯誤 {code}：{message}")
        return 3

    result = list(output)
    print("GPU 結果：", result)
    if result != expected:
        print("預期結果：", expected)
        return 4

    inverted = Array()
    code = dll.cuda_threshold_u8(source, inverted, len(values), 1, 128, 1)
    if code != 0 or list(inverted) != [255 - value for value in expected]:
        print("反相二值化測試失敗。")
        return 5

    print("測試成功。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
