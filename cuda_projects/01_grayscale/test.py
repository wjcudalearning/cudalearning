from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def main() -> int:
    dll_path = Path(__file__).resolve().parent / "01_grayscale.dll"
    if not dll_path.exists():
        print(f"找不到 DLL：{dll_path}")
        return 1

    try:
        dll = ctypes.CDLL(str(dll_path))
    except OSError as exc:
        print(f"DLL 載入失敗：{exc}")
        return 2

    u8_p = ctypes.POINTER(ctypes.c_ubyte)
    dll.cuda_bgr_to_gray_u8.argtypes = [u8_p, u8_p, ctypes.c_int, ctypes.c_int]
    dll.cuda_bgr_to_gray_u8.restype = ctypes.c_int
    dll.cuda_error_string.argtypes = [ctypes.c_int]
    dll.cuda_error_string.restype = ctypes.c_char_p

    width, height = 4, 2
    values = [
        0, 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0,
        255, 0, 0, 20, 100, 200, 50, 50, 50, 10, 20, 30,
    ]
    expected = [0, 255, 76, 150, 29, 121, 50, 22]

    InputArray = ctypes.c_ubyte * len(values)
    OutputArray = ctypes.c_ubyte * (width * height)
    source = InputArray(*values)
    output = OutputArray()

    code = dll.cuda_bgr_to_gray_u8(source, output, width, height)
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

    print("測試成功。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
