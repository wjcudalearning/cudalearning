from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def main() -> int:
    dll_path = Path(__file__).resolve().parent / "cuda_vector_add.dll"

    if not dll_path.exists():
        print(f"找不到 DLL：{dll_path}")
        return 1

    try:
        dll = ctypes.CDLL(str(dll_path))
    except OSError as exc:
        print(f"DLL 載入失敗：{exc}")
        return 2

    dll.cuda_vector_add_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    dll.cuda_vector_add_f32.restype = ctypes.c_int

    dll.cuda_error_string.argtypes = [ctypes.c_int]
    dll.cuda_error_string.restype = ctypes.c_char_p

    values_a = [1.0, 2.0, 3.0, 4.0]
    values_b = [10.0, 20.0, 30.0, 40.0]
    count = len(values_a)

    FloatArray = ctypes.c_float * count

    array_a = FloatArray(*values_a)
    array_b = FloatArray(*values_b)
    array_output = FloatArray()

    error_code = dll.cuda_vector_add_f32(
        array_a,
        array_b,
        array_output,
        count,
    )

    if error_code != 0:
        raw_message = dll.cuda_error_string(error_code)
        message = (
            raw_message.decode("utf-8", errors="replace")
            if raw_message
            else "Unknown CUDA error"
        )
        print(f"CUDA 錯誤 {error_code}：{message}")
        return 3

    result = list(array_output)
    expected = [11.0, 22.0, 33.0, 44.0]

    print("GPU 結果：", result)

    if result != expected:
        print("結果不符合預期。")
        return 4

    print("測試成功。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
