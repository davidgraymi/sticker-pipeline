"""
split_stickers.py

Detects and extracts individual stickers from a sticker sheet PNG.
Each sticker is cropped tightly to its contour — no extra padding added,
since the sticker art already includes its own white border/cut line.
Output is plain white-background PNG (no alpha manipulation needed).

Usage:
    python split_stickers.py sheet.png [--out-dir output/] [--min-area 5000]

Requirements:
    pip install opencv-python pillow numpy
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image(path: str):
    """Load image preserving alpha if present."""
    img_pil = Image.open(path).convert("RGBA")
    return img_pil, np.array(img_pil)


def find_sticker_boxes(img_np: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    """
    Find bounding boxes of each sticker on a white (or near-white) background.
    Returns list of (x, y, w, h) tuples, sorted top-left to bottom-right.
    """
    # Work on RGB channels to find non-white regions
    rgb = img_np[:, :, :3]
    # White threshold: all channels > 240
    white_mask = np.all(rgb > 240, axis=2).astype(np.uint8) * 255
    # Invert: sticker pixels are now white
    fg_mask = cv2.bitwise_not(white_mask)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))

    # Sort reading order: top-to-bottom, left-to-right (by row then column)
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def extract_stickers(
    img_pil: Image.Image,
    img_np: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    padding: int,
    out_dir: Path,
    prefix: str,
):
    """Crop each bounding box with a small padding to recover the sticker's built-in
    white border cut line, then save as a clean white-background PNG."""
    h_img, w_img = img_np.shape[:2]
    saved = []

    for i, (x, y, w, h) in enumerate(boxes, start=1):
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w_img, x + w + padding)
        y1 = min(h_img, y + h + padding)

        crop = img_pil.crop((x0, y0, x1, y1)).convert("RGB")  # keep white, no alpha tricks

        out_path = out_dir / f"{prefix}_{i:02d}.png"
        crop.save(out_path, "PNG")
        saved.append(out_path)
        print(f"  Saved: {out_path.name}  ({x1-x0}×{y1-y0}px)")

    return saved


def main():
    parser = argparse.ArgumentParser(description="Split a sticker sheet into individual PNGs.")
    parser.add_argument("sheet", help="Path to input sticker sheet image")
    parser.add_argument("--out-dir", default="stickers_out", help="Output directory (default: stickers_out)")
    parser.add_argument("--padding", type=int, default=10, help="Pixels to expand each crop to recover the sticker's built-in white border (default: 10)")
    parser.add_argument("--min-area", type=int, default=5000, help="Min contour area to consider a sticker (default: 5000)")
    parser.add_argument("--prefix", default="sticker", help="Output filename prefix (default: sticker)")
    args = parser.parse_args()

    sheet_path = Path(args.sheet)
    if not sheet_path.exists():
        print(f"Error: file not found: {sheet_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {sheet_path}")
    img_pil, img_np = load_image(str(sheet_path))
    print(f"Image size: {img_pil.width}×{img_pil.height}px")

    print("Detecting stickers...")
    boxes = find_sticker_boxes(img_np, min_area=args.min_area)
    print(f"Found {len(boxes)} sticker(s)")

    if not boxes:
        print("No stickers detected. Try lowering --min-area or check the image has a white background.")
        sys.exit(1)

    print(f"Extracting to: {out_dir}/")
    saved = extract_stickers(img_pil, img_np, boxes, args.padding, out_dir, args.prefix)

    print(f"\nDone — {len(saved)} sticker(s) extracted.")

if __name__ == "__main__":
    main()
