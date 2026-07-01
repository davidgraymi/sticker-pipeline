"""
main.py

Detects and extracts individual stickers from a sticker sheet PNG.

Pipeline:
  1. Threshold  — pixels darker than --bg-threshold are "sticker content"
  2. Grid split — projection profiles find row/column cut lines (handles
                  stickers that are too close to separate by morphology alone)
  3. Per-cell   — morphological close + contour fill within each isolated cell
  4. Crop+alpha — export each sticker as a transparent PNG with soft edges

Usage:
    python main.py sheet.png [options]

Requirements:
    pip install opencv-python pillow numpy
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _save(name: str, img: np.ndarray, debug_dir: Path | None) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / name), img)


def _save_pil(name: str, img: Image.Image, debug_dir: Path | None) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    img.save(debug_dir / name)


def _boxes_overlay(img_np: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    overlay = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2BGR).copy()
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(overlay, str(i), (x + 4, y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return overlay


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str) -> tuple[Image.Image, np.ndarray]:
    img_pil = Image.open(path).convert("RGBA")
    return img_pil, np.array(img_pil)


# ---------------------------------------------------------------------------
# Grid detection via projection profiles
# ---------------------------------------------------------------------------

def _find_grid_cuts(profile: np.ndarray, n_cuts: int, smooth_window: int = 15) -> list[int]:
    """Return n_cuts valley positions in a 1-D projection profile.

    Divides the profile into n_cuts+1 equal-width bands and finds the minimum
    in each inter-band region.  Works reliably for regular grids where each
    band contains roughly the same amount of content.
    """
    n = len(profile)
    kernel = np.ones(smooth_window) / smooth_window
    smooth = np.convolve(profile.astype(float), kernel, mode='same')

    section = n // (n_cuts + 1)
    cuts: list[int] = []
    for i in range(1, n_cuts + 1):
        lo = max(0, i * section - section // 2)
        hi = min(n, i * section + section // 2)
        local_min = lo + int(np.argmin(smooth[lo:hi]))
        cuts.append(local_min)
    return cuts


# ---------------------------------------------------------------------------
# Core segmentation
# ---------------------------------------------------------------------------

def segment_stickers(
    img_np: np.ndarray,
    bg_threshold: int,
    close_size: int,
    n_rows: int,
    n_cols: int,
    min_area: int,
    cell_overlap: int,
    debug_dir: Path | None,
) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
    """Return (boxes, masks) — one per detected sticker, reading order (top→bottom, left→right)."""

    h_img, w_img = img_np.shape[:2]

    # Step 1 — grayscale + threshold
    gray = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2GRAY)
    _save("01_grayscale.png", gray, debug_dir)

    _, thresh = cv2.threshold(gray, bg_threshold, 255, cv2.THRESH_BINARY_INV)
    _save("02_threshold.png", thresh, debug_dir)

    # Step 2 — find grid cut positions from projection profiles
    h_proj = thresh.sum(axis=1)   # sum per row
    v_proj = thresh.sum(axis=0)   # sum per column

    row_cuts = [0] + _find_grid_cuts(h_proj, n_rows - 1) + [h_img]
    col_cuts = [0] + _find_grid_cuts(v_proj, n_cols - 1) + [w_img]

    if debug_dir is not None:
        grid_viz = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2BGR).copy()
        for y in row_cuts[1:-1]:
            cv2.line(grid_viz, (0, y), (w_img, y), (0, 0, 255), 3)
        for x in col_cuts[1:-1]:
            cv2.line(grid_viz, (x, 0), (x, h_img), (255, 0, 0), 3)
        _save("03_grid.png", grid_viz, debug_dir)

    # Step 3 — per-cell: flood-fill the sheet background from the cell perimeter.
    # The shadow ring blocks the flood fill; anything it can't reach is sticker content.
    # A small close on the dark-pixel mask seals hairline gaps in the shadow ring first.
    gap_seal_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))

    boxes: list[tuple[int, int, int, int]] = []
    masks: list[np.ndarray] = []

    for row in range(n_rows):
        cy0, cy1 = row_cuts[row], row_cuts[row + 1]
        for col in range(n_cols):
            cx0, cx1 = col_cuts[col], col_cuts[col + 1]

            # No cell_overlap needed — grid cells already perfectly bound each sticker.
            cell_gray = gray[cy0:cy1, cx0:cx1]

            # Dark pixels (shadow + content). Small close seals shadow-ring gaps.
            cell_dark = (cell_gray < bg_threshold).astype(np.uint8) * 255
            cell_dark_sealed = cv2.morphologyEx(cell_dark, cv2.MORPH_CLOSE, gap_seal_k)

            # near-white mask: pixels the flood fill is allowed to travel through
            near_white = cv2.bitwise_not(cell_dark_sealed)

            # Pad with 1-px white border so the seed (0,0) is always exterior background.
            padded = cv2.copyMakeBorder(near_white, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
            flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8)
            cv2.floodFill(padded, flood_mask, (0, 0), 128)   # exterior background → 128
            flooded = padded[1:-1, 1:-1]

            # Background = near-white pixels reached by flood fill (marked 128).
            # Sticker   = everything else (dark pixels + enclosed near-white interior).
            bg_reached = (flooded == 128).astype(np.uint8) * 255
            cell_filled = cv2.bitwise_not(bg_reached)

            # Bounding box of the sticker content in this cell
            ys, xs = np.where(cell_filled > 0)
            if len(xs) == 0 or (cell_filled > 0).sum() < min_area:
                continue
            lx_cell, ly_cell = int(xs.min()), int(ys.min())
            lw = int(xs.max()) - lx_cell + 1
            lh = int(ys.max()) - ly_cell + 1

            # Place cell mask into full-image coordinates
            full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            full_mask[cy0:cy1, cx0:cx1] = cell_filled

            lx = lx_cell + cx0
            ly = ly_cell + cy0

            boxes.append((lx, ly, lw, lh))
            masks.append(full_mask)

    if debug_dir is not None:
        combined = np.zeros((h_img, w_img), dtype=np.uint8)
        for m in masks:
            combined = cv2.bitwise_or(combined, m)
        _save("04_filled.png", combined, debug_dir)

        rng = np.random.default_rng(42)
        label_viz = np.zeros((h_img, w_img, 3), dtype=np.uint8)
        for m in masks:
            color = rng.integers(80, 255, size=3).tolist()
            label_viz[m > 0] = color
        _save("05_labels.png", label_viz, debug_dir)

        _save("06_boxes.png", _boxes_overlay(img_np, boxes), debug_dir)

    return boxes, masks


# ---------------------------------------------------------------------------
# Sticker extraction
# ---------------------------------------------------------------------------

def extract_stickers(
    img_pil: Image.Image,
    img_np: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    masks: list[np.ndarray],
    padding: int,
    shadow_blur: int,
    out_dir: Path,
    prefix: str,
    debug_dir: Path | None,
) -> list[Path]:
    h_img, w_img = img_np.shape[:2]
    # blur_k=1 is a no-op (1×1 kernel); 0 is treated as "no blur"
    blur_k = max(1, shadow_blur if shadow_blur % 2 == 1 else shadow_blur + 1)
    saved: list[Path] = []

    for i, ((x, y, bw, bh), full_mask) in enumerate(zip(boxes, masks), start=1):
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w_img, x + bw + padding)
        y1 = min(h_img, y + bh + padding)

        crop_rgba = img_pil.crop((x0, y0, x1, y1)).convert("RGBA")

        alpha = full_mask[y0:y1, x0:x1].copy()
        if blur_k > 1:
            alpha = cv2.GaussianBlur(alpha, (blur_k, blur_k), 0)

        if debug_dir is not None:
            tag = f"sticker_{i:02d}"
            crop_bgr = cv2.cvtColor(np.array(crop_rgba)[:, :, :3], cv2.COLOR_RGB2BGR)
            _save(f"{tag}_a_crop.png", crop_bgr, debug_dir)
            _save(f"{tag}_b_mask.png", alpha, debug_dir)

        r, g, b, _ = crop_rgba.split()
        result = Image.merge("RGBA", (r, g, b, Image.fromarray(alpha)))

        out_path = out_dir / f"{prefix}_{i:02d}.png"
        result.save(out_path, "PNG")
        saved.append(out_path)

        if debug_dir is not None:
            _save_pil(f"sticker_{i:02d}_c_result.png", result, debug_dir)

        print(f"  [{i:02d}/{len(boxes)}] {out_path.name}  ({x1-x0}×{y1-y0}px)")

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a sticker sheet into individual transparent PNGs."
    )
    parser.add_argument("sheet", help="Path to input sticker sheet image")
    parser.add_argument("--out-dir", default="stickers_out")
    parser.add_argument("--debug", action="store_true", help="Save pipeline debug images")
    parser.add_argument("--debug-dir", default="debug")
    parser.add_argument("--n-rows", type=int, default=5,
                        help="Number of sticker rows in the sheet (default 5)")
    parser.add_argument("--n-cols", type=int, default=3,
                        help="Number of sticker columns in the sheet (default 3)")
    parser.add_argument("--bg-threshold", type=int, default=250,
                        help="Grayscale value above which a pixel is background white (default 250). "
                             "Lower to capture more shadow; raise if content is clipped.")
    parser.add_argument("--close-size", type=int, default=80,
                        help="Morphological close kernel diameter in pixels (default 30). "
                             "Raise to bridge larger intra-sticker gaps; lower if mask bleeds outside sticker.")
    parser.add_argument("--shadow-blur", type=int, default=0,
                        help="Gaussian blur kernel for alpha edge softening (default 0 = hard edge). "
                             "Only set >0 if you want a feathered edge and content never touches the mask boundary.")
    parser.add_argument("--padding", type=int, default=25,
                        help="Extra pixels around each sticker in output (default 25).")
    parser.add_argument("--min-area", type=int, default=5000,
                        help="Minimum sticker blob area in pixels (default 5000).")
    parser.add_argument("--cell-overlap", type=int, default=50,
                        help="Pixels to expand each grid cell beyond its boundary when extracting "
                             "content (default 50). Captures floating elements like decorative marks "
                             "that sit just outside the nominal cell edge.")
    parser.add_argument("--prefix", default="sticker")
    args = parser.parse_args()

    sheet_path = Path(args.sheet)
    if not sheet_path.exists():
        print(f"Error: file not found: {sheet_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(args.debug_dir) if args.debug else None

    print(f"Loading: {sheet_path}")
    img_pil, img_np = load_image(str(sheet_path))
    print(f"Image size: {img_pil.width}×{img_pil.height}px")

    print(f"Segmenting ({args.n_rows}×{args.n_cols} grid, "
          f"bg-threshold={args.bg_threshold}, close-size={args.close_size})...")
    boxes, masks = segment_stickers(
        img_np,
        bg_threshold=args.bg_threshold,
        close_size=args.close_size,
        n_rows=args.n_rows,
        n_cols=args.n_cols,
        min_area=args.min_area,
        cell_overlap=args.cell_overlap,
        debug_dir=debug_dir,
    )
    print(f"Found {len(boxes)} sticker(s)")

    if not boxes:
        print("No stickers detected. Try lowering --bg-threshold or --min-area.")
        sys.exit(1)

    print(f"Extracting to: {out_dir}/")
    saved = extract_stickers(
        img_pil, img_np, boxes, masks,
        padding=args.padding,
        shadow_blur=args.shadow_blur,
        out_dir=out_dir,
        prefix=args.prefix,
        debug_dir=debug_dir,
    )

    if debug_dir:
        print(f"Debug images → {debug_dir}/")

    print(f"\nDone — {len(saved)} sticker(s) saved to {out_dir}/")


if __name__ == "__main__":
    main()
