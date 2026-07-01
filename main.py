"""
main.py

Detects and extracts individual stickers from a sticker sheet PNG using
contour-based segmentation, then builds a per-sticker alpha mask via
morphological operations to preserve all sticker content (artwork + floating
text) and remove the white sheet background.

Usage:
    python main.py sheet.png [--out-dir output/] [--min-area 5000]

Requirements:
    pip install opencv-python pillow numpy
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image(path: str) -> tuple[Image.Image, np.ndarray]:
    img_pil = Image.open(path).convert("RGBA")
    return img_pil, np.array(img_pil)


def compute_fg_mask(img_np: np.ndarray) -> np.ndarray:
    """Binary mask: 255 = non-white (sticker content), 0 = white background."""
    rgb = img_np[:, :, :3]
    white = np.all(rgb > 240, axis=2).astype(np.uint8) * 255
    fg = cv2.bitwise_not(white)
    # Light denoise only — heavy closing happens per-crop to preserve text
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=1)


def find_sticker_boxes(
    fg_mask: np.ndarray, min_area: int
) -> list[tuple[int, int, int, int]]:
    """
    Find bounding boxes of individual stickers by running heavier morphology on
    a copy of the mask (to merge nearby blobs belonging to the same sticker).
    Returns (x, y, w, h) list sorted reading order.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    merged = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k, iterations=3)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def build_sticker_mask(crop_fg: np.ndarray, close_px: int, border_px: int) -> np.ndarray:
    """
    Build a per-sticker alpha mask from the raw foreground crop:
      1. Large closing: joins floating text + sparkles + cow into one region.
      2. Dilation: recovers the white die-cut border.
      3. Gaussian blur: softens edges for clean transparency.
    """
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1)
    )
    mask = cv2.morphologyEx(crop_fg, cv2.MORPH_CLOSE, k_close)

    k_border = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (border_px * 2 + 1, border_px * 2 + 1)
    )
    mask = cv2.dilate(mask, k_border)

    return cv2.GaussianBlur(mask, (5, 5), 0)


def extract_stickers(
    img_pil: Image.Image,
    img_np: np.ndarray,
    fg_mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    padding: int,
    close_px: int,
    border_px: int,
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    h_img, w_img = img_np.shape[:2]
    saved = []

    for i, (x, y, w, h) in enumerate(boxes, start=1):
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w_img, x + w + padding)
        y1 = min(h_img, y + h + padding)

        crop_rgba = img_pil.crop((x0, y0, x1, y1)).convert("RGBA")
        crop_fg = fg_mask[y0:y1, x0:x1]

        alpha = build_sticker_mask(crop_fg, close_px=close_px, border_px=border_px)

        r, g, b, _ = crop_rgba.split()
        result = Image.merge("RGBA", (r, g, b, Image.fromarray(alpha)))

        out_path = out_dir / f"{prefix}_{i:02d}.png"
        result.save(out_path, "PNG")
        saved.append(out_path)
        print(f"  [{i:02d}/{len(boxes)}] {out_path.name}  ({x1-x0}×{y1-y0}px)")

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Split a sticker sheet into individual transparent PNGs."
    )
    parser.add_argument("sheet", help="Path to input sticker sheet image")
    parser.add_argument("--out-dir", default="stickers_out", help="Output directory (default: stickers_out)")
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Pixels to expand each crop beyond the detected bounding box (default: 20)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=5000,
        help="Min contour area in pixels to count as a sticker (default: 5000)",
    )
    parser.add_argument(
        "--close-px",
        type=int,
        default=80,
        help="Morphological closing radius to bridge gaps between floating text and artwork (default: 80)",
    )
    parser.add_argument(
        "--border-px",
        type=int,
        default=18,
        help="Dilation radius to recover the sticker's white die-cut border (default: 18)",
    )
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

    print("Computing foreground mask...")
    fg_mask = compute_fg_mask(img_np)

    print("Segmenting stickers via contour detection...")
    boxes = find_sticker_boxes(fg_mask, min_area=args.min_area)
    print(f"Found {len(boxes)} sticker(s)")

    if not boxes:
        print("No stickers detected. Try lowering --min-area or check the image has a white background.")
        sys.exit(1)

    print(f"Extracting with transparent backgrounds to: {out_dir}/")
    saved = extract_stickers(
        img_pil,
        img_np,
        fg_mask,
        boxes,
        padding=args.padding,
        close_px=args.close_px,
        border_px=args.border_px,
        out_dir=out_dir,
        prefix=args.prefix,
    )

    print(f"\nDone — {len(saved)} sticker(s) saved to {out_dir}/")


if __name__ == "__main__":
    main()
