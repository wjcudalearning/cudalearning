from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def cpu_gaussian(values: list[int], width: int, height: int) -> list[int]:
    weights = ((1, 2, 1), (2, 4, 2), (1, 2, 1))
    result: list[int] = []
    for y in range(height):
        for x in range(width):
            total = 0
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    sx = min(max(x + kx, 0), width - 1)
                    sy = min(max(y + ky, 0), height - 1)
                    total += values[sy * width + sx] * weights[ky + 1][kx + 1]
            result.append((total + 8) // 16)
    return result


def main() -> int:
    dll_path = Path(__file__).resolve().parent / "03_gaussian_blur.dll"
    if not dll_path.exists():
        print(f"找不到 DLL：{dll_path}")
        return 1

    try:
        dll = ctypes.CDLL(str(dll_path))
    except OSError as exc:
        print(f"DLL 載入失敗：{exc}")
        return 2

    u8_p = ctypes.POINTER(ctypes.c_ubyte)
    dll.cuda_gaussian_blur_3x3_u8.argtypes = [u8_p, u8_p, ctypes.c_int, ctypes.c_int]
    dll.cuda_gaussian_blur_3x3_u8.restype = ctypes.c_int
    dll.cuda_error_string.argtypes = [ctypes.c_int]
    dll.cuda_error_string.restype = ctypes.c_char_p

    width, height = 8, 8
    values = [0] * (width * height)
    values[(height // 2) * width + width // 2] = 255
    expected = cpu_gaussian(values, width, height)

    Array = ctypes.c_ubyte * len(values)
    source = Array(*values)
    output = Array()
    code = dll.cuda_gaussian_blur_3x3_u8(source, output, width, height)
    if code != 0:
        raw = dll.cuda_error_string(code)
        message = raw.decode("utf-8", errors="replace") if raw else "Unknown CUDA error"
        print(f"CUDA 錯誤 {code}：{message}")
        return 3

    result = list(output)
    if result != expected:
        print("GPU 與 CPU 結果不同。")
        print("GPU：", result)
        print("CPU：", expected)
        return 4

    print("中心 3x3：")
    for y in range(3, 6):
        print(result[y * width + 3 : y * width + 6])
    print("測試成功。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
