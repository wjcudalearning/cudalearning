from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "demo_data"
    output_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(20260713)
    image_array = rng.normal(80, 5, size=(700, 1000)).clip(0, 255).astype(np.uint8)

    template = Image.new("L", (72, 54), 45)
    draw = ImageDraw.Draw(template)
    draw.rectangle((4, 4, 67, 49), outline=210, width=3)
    draw.ellipse((13, 10, 42, 39), fill=135, outline=235, width=2)
    draw.line((8, 45, 62, 8), fill=250, width=4)
    draw.rectangle((47, 29, 64, 46), fill=185)

    template_array = np.asarray(template, dtype=np.uint8)
    positions = [(80, 90), (330, 130), (680, 75), (190, 430), (610, 470)]
    brightness_offsets = [0, 12, -10, 20, -18]

    for (x, y), offset in zip(positions, brightness_offsets):
        patch = np.clip(template_array.astype(np.int16) + offset, 0, 255).astype(np.uint8)
        image_array[y : y + patch.shape[0], x : x + patch.shape[1]] = patch

    image_path = output_dir / "demo_large.png"
    template_path = output_dir / "demo_template.png"
    Image.fromarray(image_array, mode="L").save(image_path)
    template.save(template_path)

    print(f"大圖：{image_path}")
    print(f"Template：{template_path}")
    print(f"預期位置：{positions}")


if __name__ == "__main__":
    main()
