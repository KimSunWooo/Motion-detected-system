#!/usr/bin/env python3
"""Export dashboard skeleton scenario animations as GIF files.

Matches the canvas renderer in templates/index.html (normalized coords,
head bbox / ear circles / bones / wrist highlights).

Usage:
  PYTHONPATH=src python scripts/export_scenario_gifs.py
  PYTHONPATH=src python scripts/export_scenario_gifs.py --out outputs/gifs --fps 14
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helmet_action.visualization.demo import build_demo_scenarios  # noqa: E402


STATE_COLORS = {
    "idle": (138, 160, 196),
    "bbox": (61, 214, 140),
    "ear_grasp": (255, 143, 171),
    "scratch": (61, 214, 140),
    "helmet_off": (255, 92, 92),
}


def _slug(name: str, idx: int) -> str:
    # ASCII-friendly filename from Korean scenario title
    mapping = {
        "하이 앵글 · 단순 머리 긁기": "01_highangle_scratch",
        "하이 앵글 · 안전모 벗기": "02_highangle_helmet_off",
        "원거리 축소 카메라 · 긁기": "03_far_scratch",
        "근거리 확대 카메라 · 벗기": "04_near_helmet_off",
        "팔만 흔들기 · 머리 비접촉": "05_idle_no_contact",
    }
    if name in mapping:
        return mapping[name]
    safe = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE).strip("_")
    return f"{idx:02d}_{safe or 'scenario'}"


def render_frame(sc: dict, frame_i: int, *, width: int = 720, height: int = 860) -> np.ndarray:
    """Return RGB uint8 image matching dashboard canvas draw()."""
    from PIL import Image, ImageDraw, ImageFont

    fr = sc["frames"][frame_i]
    k = fr["keypoints"]
    bones = sc["bones"]

    img = Image.new("RGB", (width, height), (11, 18, 32))
    draw = ImageDraw.Draw(img, "RGBA")

    sx = width / 2
    sy = 250
    s = 240.0

    def X(x: float) -> float:
        return sx + float(x) * s

    def Y(y: float) -> float:
        return sy + float(y) * s

    x0, y0, x1, y1 = fr["bbox"]
    bx0, by0, bx1, by1 = X(x0), Y(y0), X(x1), Y(y1)
    draw.rectangle([bx0, by0, bx1, by1], fill=(245, 197, 24, 26), outline=(245, 197, 24, 255), width=2)

    # ear / center circles
    cx, cy = fr["center"]
    cr = float(fr["center_r"]) * s
    draw.ellipse([X(cx) - cr, Y(cy) - cr, X(cx) + cr, Y(cy) + cr], outline=(61, 214, 140, 255), width=2)
    for ear_key in ("left_ear", "right_ear"):
        ex, ey = fr[ear_key]
        er = float(fr["ear_r"]) * s
        draw.ellipse([X(ex) - er, Y(ey) - er, X(ex) + er, Y(ey) + er], outline=(255, 143, 171, 255), width=2)

    # bones
    for a, b in bones:
        draw.line([(X(k[a][0]), Y(k[a][1])), (X(k[b][0]), Y(k[b][1]))], fill=(94, 224, 255, 255), width=4)
    # neck bridge shoulders → origin
    draw.line([(X(k[5][0]), Y(k[5][1])), (X(0), Y(0)), (X(k[6][0]), Y(k[6][1]))], fill=(94, 224, 255, 255), width=4)

    for p in range(len(k)):
        color = (255, 209, 102) if p in (9, 10) else (215, 236, 255)
        r = 5
        draw.ellipse([X(k[p][0]) - r, Y(k[p][1]) - r, X(k[p][0]) + r, Y(k[p][1]) + r], fill=color)
    draw.ellipse([X(0) - 6, Y(0) - 6, X(0) + 6, Y(0) + 6], fill=(124, 255, 178))

    state = str(fr.get("state", "idle"))
    scolor = STATE_COLORS.get(state, (138, 160, 196))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        font_s = font

    draw.text((24, 20), f"state: {state}", fill=scolor + (255,), font=font)
    draw.text(
        (24, 48),
        "high-angle upper body · neck=(0,0) · shoulder width=1.0",
        fill=(138, 160, 196, 255),
        font=font_s,
    )
    draw.text((24, height - 36), f"{sc['name']}  ·  frame {frame_i}/{sc['n_frames'] - 1}", fill=(170, 185, 210, 255), font=font_s)

    return np.asarray(img.convert("RGB"))


def export_scenario_gif(
    sc: dict,
    out_path: Path,
    *,
    fps: float = 14.0,
    width: int = 720,
    height: int = 860,
    every_n: int = 1,
) -> Path:
    from PIL import Image

    frames_rgb = []
    for i in range(0, int(sc["n_frames"]), max(1, every_n)):
        frames_rgb.append(Image.fromarray(render_frame(sc, i, width=width, height=height)))

    if not frames_rgb:
        raise RuntimeError(f"no frames for scenario {sc.get('name')}")

    duration_ms = int(round(1000.0 / max(fps, 1e-3)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_rgb[0].save(
        out_path,
        save_all=True,
        append_images=frames_rgb[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export dashboard skeleton scenarios as GIFs")
    parser.add_argument("--out", default="outputs/gifs", help="output directory")
    parser.add_argument("--fps", type=float, default=14.0, help="GIF playback FPS (dashboard ~14)")
    parser.add_argument("--every-n", type=int, default=1, help="keep every Nth frame")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=860)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    scenarios = build_demo_scenarios()
    written: list[Path] = []
    for i, sc in enumerate(scenarios):
        slug = _slug(sc["name"], i)
        path = out_dir / f"{slug}.gif"
        export_scenario_gif(
            sc,
            path,
            fps=args.fps,
            width=args.width,
            height=args.height,
            every_n=args.every_n,
        )
        written.append(path)
        print(f"wrote {path}  ({sc['n_frames']} frames, {sc['name']})")

    # also write an index text for convenience
    index = out_dir / "README.txt"
    lines = [
        "Dashboard skeleton scenario GIFs",
        f"Source: http://127.0.0.1:8765/  (same sequences as /api/scenarios)",
        f"FPS: {args.fps}",
        "",
    ]
    for p, sc in zip(written, scenarios):
        lines.append(f"{p.name}\t{sc['name']}\t{sc['synthetic_scenario']}\t{sc['expected']}")
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"index: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
