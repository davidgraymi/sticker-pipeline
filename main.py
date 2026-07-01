"""
main.py

Detects and extracts individual stickers from a sticker sheet PNG using
bubble-edge segmentation. Supports three edge detectors:
  - Canny (fast, classical)
  - HED  — Holistically-Nested Edge Detection (deep learning, ~56 MB model)
  - TEED — Tiny and Efficient Edge Detector (deep learning, ~6 MB model)

Pass --compare to run all three in debug mode and save side-by-side comparison images.
HED is used for segmentation when available; otherwise TEED, then Canny.

Usage:
    python main.py sheet.png [--out-dir output/] [--min-area 5000] [--debug] [--compare]

Requirements:
    pip install opencv-python pillow numpy torch
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from teed import TED, load_teed_net, run_teed


class _CropLayer:
    """Custom Caffe crop layer for OpenCV DNN — required by HED."""

    def __init__(self, _params, _blobs):
        self.x0 = self.y0 = self.x1 = self.y1 = 0

    def getMemoryShapes(self, inputs):
        src, ref = inputs[0], inputs[1]
        self.y0 = (src[2] - ref[2]) // 2
        self.x0 = (src[3] - ref[3]) // 2
        self.y1 = self.y0 + ref[2]
        self.x1 = self.x0 + ref[3]
        return [[src[0], src[1], ref[2], ref[3]]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.y0:self.y1, self.x0:self.x1]]


def load_hed_net(model_dir: Path) -> cv2.dnn.Net:
    model_dir.mkdir(parents=True, exist_ok=True)

    proto_path = model_dir / "hed_deploy.prototxt"
    model_path = model_dir / "hed_pretrained_bsds.caffemodel"

    cv2.dnn_registerLayer("Crop", _CropLayer)
    net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
    return net


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def save_debug(name: str, img: np.ndarray, debug_dir: Path | None) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / name), img)


def save_debug_pil(name: str, img: Image.Image, debug_dir: Path | None) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    img.save(debug_dir / name)


def draw_boxes_overlay(img_np: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    overlay = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2BGR).copy()
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(overlay, str(i), (x + 4, y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return overlay


def labeled_panel(img_gray: np.ndarray, label: str, font_scale: float = 1.2) -> np.ndarray:
    """Convert a grayscale image to BGR and stamp a label at the top."""
    panel = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(panel, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 200, 255), 2, cv2.LINE_AA)
    return panel


def side_by_side(images: list[np.ndarray], gap: int = 8) -> np.ndarray:
    """Stack BGR images horizontally with a thin black gap."""
    h = max(im.shape[0] for im in images)
    padded = []
    for im in images:
        if im.shape[0] < h:
            pad = np.zeros((h - im.shape[0], im.shape[1], 3), dtype=np.uint8)
            im = np.vstack([im, pad])
        padded.append(im)
        padded.append(np.zeros((h, gap, 3), dtype=np.uint8))
    return np.hstack(padded[:-1])


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str) -> tuple[Image.Image, np.ndarray]:
    img_pil = Image.open(path).convert("RGBA")
    return img_pil, np.array(img_pil)


# ---------------------------------------------------------------------------
# Step 1 — grayscale
# ---------------------------------------------------------------------------

def to_grayscale(img_np: np.ndarray, debug_dir: Path | None) -> np.ndarray:
    gray = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2GRAY)
    save_debug("01_grayscale.png", gray, debug_dir)
    return gray


# ---------------------------------------------------------------------------
# Step 2a — Canny edge detection
# ---------------------------------------------------------------------------

def detect_edges_canny(
    gray: np.ndarray,
    blur_ksize: int,
    canny_low: int,
    canny_high: int,
    debug_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    save_debug("02_canny_blurred.png", blurred, debug_dir)

    edges = cv2.Canny(blurred, canny_low, canny_high)
    save_debug("03a_canny_edges_raw.png", edges, debug_dir)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    save_debug("04a_canny_edges_closed.png", edges_closed, debug_dir)

    return edges, edges_closed


# ---------------------------------------------------------------------------
# Step 2b — HED edge detection
# ---------------------------------------------------------------------------

def detect_edges_hed(
    img_np: np.ndarray,
    net: cv2.dnn.Net,
    debug_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = img_np.shape[:2]
    img_bgr = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2BGR)

    # ImageNet mean subtraction, no scaling — matches HED training
    blob = cv2.dnn.blobFromImage(
        img_bgr, scalefactor=1.0, size=(w, h),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False, crop=False,
    )
    net.setInput(blob)
    raw = net.forward("sigmoid-fuse")          # shape: (1, 1, H, W)
    edges_f = raw[0, 0]                        # float32 in [0, 1]
    edges = (edges_f * 255).clip(0, 255).astype(np.uint8)
    save_debug("03b_hed_edges_raw.png", edges, debug_dir)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    save_debug("04b_hed_edges_closed.png", edges_closed, debug_dir)

    return edges, edges_closed


# ---------------------------------------------------------------------------
# Step 2c — TEED edge detection
# ---------------------------------------------------------------------------

def detect_edges_teed(
    img_np: np.ndarray,
    model: TED,
    debug_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    edges = run_teed(model, img_np)
    save_debug("03c_teed_edges_raw.png", edges, debug_dir)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    save_debug("04c_teed_edges_closed.png", edges_closed, debug_dir)

    return edges, edges_closed


# ---------------------------------------------------------------------------
# Step 2d — side-by-side comparison debug image
# ---------------------------------------------------------------------------

def save_edge_comparison(
    canny_raw: np.ndarray,
    canny_closed: np.ndarray,
    hed_raw: np.ndarray | None,
    hed_closed: np.ndarray | None,
    teed_raw: np.ndarray | None,
    teed_closed: np.ndarray | None,
    debug_dir: Path | None,
) -> None:
    if debug_dir is None:
        return
    panels = [
        labeled_panel(canny_raw,    "Canny — raw"),
        labeled_panel(canny_closed, "Canny — closed"),
    ]
    if hed_raw is not None and hed_closed is not None:
        panels += [
            labeled_panel(hed_raw,      "HED   — raw"),
            labeled_panel(hed_closed,   "HED   — closed"),
        ]
    if teed_raw is not None and teed_closed is not None:
        panels += [
            labeled_panel(teed_raw,    "TEED  — raw"),
            labeled_panel(teed_closed, "TEED  — closed"),
        ]
    save_debug("05_edge_comparison.png", side_by_side(panels), debug_dir)


# ---------------------------------------------------------------------------
# Step 3 — fill bubble contours → sticker bounding boxes + per-sticker masks
# ---------------------------------------------------------------------------

def find_bubble_contours(
    edges_closed: np.ndarray,
    img_np: np.ndarray,
    min_area: int,
    step_prefix: str,
    debug_dir: Path | None,
) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    h, w = edges_closed.shape
    contour_viz = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2BGR).copy()
    cv2.drawContours(contour_viz, contours, -1, (0, 255, 0), 2)
    save_debug(f"{step_prefix}_contours.png", contour_viz, debug_dir)

    bubble_fill = np.zeros((h, w), dtype=np.uint8)
    boxes: list[tuple[int, int, int, int]] = []
    masks: list[np.ndarray] = []

    for c in contours:
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(m, [c], -1, 255, cv2.FILLED)
        masks.append(m)
        x, y, bw, bh = cv2.boundingRect(c)
        boxes.append((x, y, bw, bh))
        cv2.drawContours(bubble_fill, [c], -1, 255, cv2.FILLED)

    paired = sorted(zip(boxes, masks), key=lambda p: (p[0][1], p[0][0]))
    boxes = [p[0] for p in paired]
    masks = [p[1] for p in paired]

    save_debug(f"{step_prefix}_fills.png", bubble_fill, debug_dir)

    box_overlay = draw_boxes_overlay(img_np, boxes)
    save_debug(f"{step_prefix}_boxes.png", box_overlay, debug_dir)

    return boxes, masks


def save_fill_comparison(
    canny_fill: np.ndarray,
    hed_fill: np.ndarray | None,
    teed_fill: np.ndarray | None,
    debug_dir: Path | None,
) -> None:
    if debug_dir is None:
        return
    panels = [labeled_panel(canny_fill, "Canny fills")]
    if hed_fill is not None:
        panels.append(labeled_panel(hed_fill, "HED fills"))
    if teed_fill is not None:
        panels.append(labeled_panel(teed_fill, "TEED fills"))
    save_debug("08_fill_comparison.png", side_by_side(panels), debug_dir)


# ---------------------------------------------------------------------------
# Step 4 — extract each sticker using its bubble mask
# ---------------------------------------------------------------------------

def extract_stickers(
    img_pil: Image.Image,
    img_np: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    bubble_masks: list[np.ndarray],
    padding: int,
    out_dir: Path,
    prefix: str,
    debug_dir: Path | None,
) -> list[Path]:
    h_img, w_img = img_np.shape[:2]
    saved: list[Path] = []

    for i, ((x, y, w, h), full_mask) in enumerate(zip(boxes, bubble_masks), start=1):
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w_img, x + w + padding)
        y1 = min(h_img, y + h + padding)

        crop_rgba = img_pil.crop((x0, y0, x1, y1)).convert("RGBA")
        alpha = full_mask[y0:y1, x0:x1]
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

        if debug_dir is not None:
            tag = f"sticker_{i:02d}"
            crop_bgr = cv2.cvtColor(np.array(crop_rgba)[:, :, :3], cv2.COLOR_RGB2BGR)
            save_debug(f"{tag}_a_crop.png", crop_bgr, debug_dir)
            save_debug(f"{tag}_b_mask.png", alpha, debug_dir)

        r, g, b, _ = crop_rgba.split()
        result = Image.merge("RGBA", (r, g, b, Image.fromarray(alpha)))

        out_path = out_dir / f"{prefix}_{i:02d}.png"
        result.save(out_path, "PNG")
        saved.append(out_path)

        if debug_dir is not None:
            save_debug_pil(f"sticker_{i:02d}_c_result.png", result, debug_dir)

        print(f"  [{i:02d}/{len(boxes)}] {out_path.name}  ({x1-x0}×{y1-y0}px)")

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a sticker sheet into individual transparent PNGs using bubble-edge segmentation."
    )
    parser.add_argument("sheet", help="Path to input sticker sheet image")
    parser.add_argument("--out-dir", default="stickers_out")
    parser.add_argument("--debug", action="store_true", help="Save pipeline debug images to --debug-dir")
    parser.add_argument("--debug-dir", default="debug")
    parser.add_argument("--compare", action="store_true",
                        help="Run HED and TEED alongside Canny and save side-by-side comparisons (requires --debug; downloads models on first run)")
    parser.add_argument("--models-dir", default="models", help="Directory to cache HED model files")
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--min-area", type=int, default=5000)
    parser.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel for Canny (must be odd)")
    parser.add_argument("--canny-low", type=int, default=30)
    parser.add_argument("--canny-high", type=int, default=80)
    parser.add_argument("--prefix", default="sticker")
    args = parser.parse_args()

    sheet_path = Path(args.sheet)
    if not sheet_path.exists():
        print(f"Error: file not found: {sheet_path}", file=sys.stderr)
        sys.exit(1)

    blur_ksize = args.blur if args.blur % 2 == 1 else args.blur + 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(args.debug_dir) if args.debug else None

    print(f"Loading: {sheet_path}")
    img_pil, img_np = load_image(str(sheet_path))
    print(f"Image size: {img_pil.width}×{img_pil.height}px")

    print("Step 1 — grayscale...")
    gray = to_grayscale(img_np, debug_dir)

    print(f"Step 2a — Canny edges (blur={blur_ksize}, thresholds={args.canny_low}/{args.canny_high})...")
    canny_raw, canny_closed = detect_edges_canny(gray, blur_ksize, args.canny_low, args.canny_high, debug_dir)

    hed_raw = hed_closed = None
    teed_raw = teed_closed = None
    if args.compare:
        if not args.debug:
            print("Warning: --compare has no effect without --debug. Add --debug to save comparison images.")
        else:
            print("Step 2b — HED edges (loading model)...")
            net = load_hed_net(Path(args.models_dir))
            hed_raw, hed_closed = detect_edges_hed(img_np, net, debug_dir)

            print("Step 2c — TEED edges (loading model)...")
            teed_model = load_teed_net(Path(args.models_dir))
            teed_raw, teed_closed = detect_edges_teed(img_np, teed_model, debug_dir)

            save_edge_comparison(canny_raw, canny_closed, hed_raw, hed_closed, teed_raw, teed_closed, debug_dir)

    # Use HED for segmentation when available, otherwise TEED, otherwise Canny
    if hed_closed is not None:
        active_edges, active_label = hed_closed, "HED"
    elif teed_closed is not None:
        active_edges, active_label = teed_closed, "TEED"
    else:
        active_edges, active_label = canny_closed, "Canny"

    print(f"Step 3 — bubble contours via {active_label} (min-area={args.min_area})...")
    boxes, masks = find_bubble_contours(active_edges, img_np, args.min_area, "06_active", debug_dir)
    print(f"Found {len(boxes)} sticker(s)")

    if args.compare and debug_dir is not None and (hed_closed is not None or teed_closed is not None):
        print("  (also segmenting with remaining detectors for fill comparison...)")
        h_img, w_img = img_np.shape[:2]

        # active fill (HED or TEED, whichever drove segmentation)
        active_fill = np.zeros((h_img, w_img), dtype=np.uint8)
        for m in masks:
            active_fill = cv2.bitwise_or(active_fill, m)

        _, canny_masks = find_bubble_contours(canny_closed, img_np, args.min_area, "07_canny", debug_dir)
        canny_fill = np.zeros((h_img, w_img), dtype=np.uint8)
        for m in canny_masks:
            canny_fill = cv2.bitwise_or(canny_fill, m)

        hed_fill: np.ndarray | None = None
        teed_fill: np.ndarray | None = None
        if hed_closed is not None and teed_closed is not None:
            # active was HED; also compute TEED fill
            hed_fill = active_fill
            _, teed_masks = find_bubble_contours(teed_closed, img_np, args.min_area, "07_teed", debug_dir)
            teed_fill = np.zeros((h_img, w_img), dtype=np.uint8)
            for m in teed_masks:
                teed_fill = cv2.bitwise_or(teed_fill, m)
        elif hed_closed is not None:
            hed_fill = active_fill
        else:
            teed_fill = active_fill

        save_fill_comparison(canny_fill, hed_fill, teed_fill, debug_dir)

    if not boxes:
        print("No bubbles detected. Try lowering --min-area or adjusting edge detection parameters.")
        sys.exit(1)

    print(f"Step 4 — extracting to: {out_dir}/")
    saved = extract_stickers(
        img_pil, img_np, boxes, masks,
        padding=args.padding,
        out_dir=out_dir,
        prefix=args.prefix,
        debug_dir=debug_dir,
    )

    if debug_dir:
        print(f"Debug images → {debug_dir}/")

    print(f"\nDone — {len(saved)} sticker(s) saved to {out_dir}/")


if __name__ == "__main__":
    main()
