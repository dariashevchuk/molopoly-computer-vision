from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np


@dataclass
class CalibCFG:
    video_path: str = "data/easy/2_easy.mp4"
    warp_size: int = 900

    # board quad detection
    canny1: int = 60
    canny2: int = 160
    blur_ksize: int = 7
    dilate_ksize: int = 3
    quad_area_ratio_min: float = 0.15
    approx_eps_ratio: float = 0.02
    max_contours_to_try: int = 12

    # used for homography estimation
    board_sample_frames: int = 80

    # inner box detection (Sobel profiles)
    inner_sample_frames: int = 70
    band_ratio: float = 0.30
    smooth_ksize: int = 51
    min_margin_ratio: float = 0.06

    
    show_debug: bool = True
    save_debug: bool = False
    debug_dir: str = "debug_out"


@dataclass
class CalibrationResult:
    warp_size: int
    H: np.ndarray                    # 3x3 homography
    quad_med: np.ndarray             # 4x2 (tl,tr,br,bl)
    inner_box: Tuple[int, int, int, int]  # (xl, yt, xr, yb)
    dice_roi: Tuple[int, int, int, int]   # same as inner_box by default
    fields: List[Tuple[int, int, int, int]]
    meta: Dict[str, float]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _dbg_save(cfg: CalibCFG, name: str, img: np.ndarray) -> None:
    if not cfg.save_debug:
        return
    _ensure_dir(cfg.debug_dir)
    cv2.imwrite(os.path.join(cfg.debug_dir, name), img)



def read_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read video: {video_path}")
    return frame

def sample_frames(video_path: str, n: int, stride: int = 1) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(frame)
        i += 1
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"No frames read from: {video_path}")
    return frames


# Board quad detection (edges -> contours -> 4-corner quad)
def order_quad_pts(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def find_board_quad(frame_bgr: np.ndarray, cfg: CalibCFG) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    k = cfg.blur_ksize if cfg.blur_ksize % 2 == 1 else cfg.blur_ksize + 1
    gray = cv2.GaussianBlur(gray, (k, k), 0)

    edges = cv2.Canny(gray, cfg.canny1, cfg.canny2)

    dk = cfg.dilate_ksize
    edges = cv2.dilate(edges, np.ones((dk, dk), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts[:cfg.max_contours_to_try]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, cfg.approx_eps_ratio * peri, True)
        if len(approx) != 4:
            continue
        if cv2.contourArea(approx) < cfg.quad_area_ratio_min * (h * w):
            continue
        quad = approx.reshape(-1, 2)
        return order_quad_pts(quad)

    return None


# Homography

def estimate_homography(video_path: str, cfg: CalibCFG) -> Tuple[np.ndarray, np.ndarray]:
    frames = sample_frames(video_path, n=cfg.board_sample_frames, stride=1)
    quads = []
    for f in frames:
        q = find_board_quad(f, cfg)
        if q is not None:
            quads.append(q)

    if len(quads) < 10:
        raise RuntimeError(
            "Board quad not detected reliably. Tune canny/area/approx parameters or improve lighting."
        )

    quad_med = np.median(np.stack(quads, axis=0), axis=0).astype(np.float32)
    s = cfg.warp_size
    dst = np.array([[0, 0], [s - 1, 0], [s - 1, s - 1], [0, s - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad_med, dst)
    return H, quad_med

def warp_board(frame_bgr: np.ndarray, H: np.ndarray, warp_size: int) -> np.ndarray:
    return cv2.warpPerspective(frame_bgr, H, (warp_size, warp_size))



# Inner box detection (Sobel edge profiles)

def inner_box_from_edge_profiles(
    warped_bgr: np.ndarray,
    band_ratio: float,
    smooth_ksize: int,
    min_margin_ratio: float,
) -> Optional[Tuple[int, int, int, int]]:
    s = warped_bgr.shape[0]
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    ax = np.abs(gx)
    ay = np.abs(gy)

    c = s // 2
    band = int(band_ratio * s)
    y1, y2 = max(0, c - band), min(s, c + band)
    x1, x2 = max(0, c - band), min(s, c + band)

    px = ax[y1:y2, :].mean(axis=0)
    py = ay[:, x1:x2].mean(axis=1)

    k = smooth_ksize if smooth_ksize % 2 == 1 else smooth_ksize + 1
    px = cv2.GaussianBlur(px.reshape(1, -1), (k, 1), 0).reshape(-1)
    py = cv2.GaussianBlur(py.reshape(-1, 1), (1, k), 0).reshape(-1)

    m = int(min_margin_ratio * s)
    left = np.arange(m, c - m)
    right = np.arange(c + m, s - m)
    top = np.arange(m, c - m)
    bot = np.arange(c + m, s - m)
    if len(left) == 0 or len(right) == 0 or len(top) == 0 or len(bot) == 0:
        return None

    xl = int(left[np.argmax(px[left])])
    xr = int(right[np.argmax(px[right])])
    yt = int(top[np.argmax(py[top])])
    yb = int(bot[np.argmax(py[bot])])

    # inner box box cant be tiny
    if (xr - xl) < 0.35 * s or (yb - yt) < 0.35 * s:
        return None

    return (xl, yt, xr, yb)

def estimate_inner_box_median(video_path: str, H: np.ndarray, cfg: CalibCFG) -> Tuple[int, int, int, int]:
    cap = cv2.VideoCapture(video_path)
    boxes = []
    tries = 0

    while len(boxes) < cfg.inner_sample_frames:
        ok, frame = cap.read()
        if not ok:
            break
        warped = warp_board(frame, H, cfg.warp_size)
        box = inner_box_from_edge_profiles(
            warped,
            band_ratio=cfg.band_ratio,
            smooth_ksize=cfg.smooth_ksize,
            min_margin_ratio=cfg.min_margin_ratio,
        )
        if box is not None:
            boxes.append(box)

        tries += 1
        if tries > cfg.inner_sample_frames * 5:
            break

    cap.release()
    if len(boxes) < max(10, cfg.inner_sample_frames // 4):
        raise RuntimeError("Too few inner boxes")

    b = np.array(boxes, dtype=np.float32)
    med = np.median(b, axis=0)
    return tuple(int(round(x)) for x in med)


# Fields (40 rectangles) + masks
def build_monopoly_fields_robust(warp_size: int, inner_box: Tuple[int, int, int, int]):
    s = warp_size
    x1, y1, x2, y2 = inner_box

    t_left = int(max(1, x1))
    t_top = int(max(1, y1))
    t_right = int(max(1, s - x2))
    t_bottom = int(max(1, s - y2))

    cell_x = (s - t_left - t_right) / 9.0
    cell_y = (s - t_top - t_bottom) / 9.0

    def rect(ax1, ay1, ax2, ay2):
        return (int(round(ax1)), int(round(ay1)), int(round(ax2)), int(round(ay2)))

    fields = []
    # top
    fields.append(rect(0, 0, t_left, t_top))  
    for i in range(1, 10):
        xa1 = t_left + (i - 1) * cell_x
        xa2 = t_left + i * cell_x
        fields.append(rect(xa1, 0, xa2, t_top))
    fields.append(rect(s - t_right, 0, s, t_top))  
    # right
    for i in range(1, 10):
        ya1 = t_top + (i - 1) * cell_y
        ya2 = t_top + i * cell_y
        fields.append(rect(s - t_right, ya1, s, ya2))
    fields.append(rect(s - t_right, s - t_bottom, s, s))  # corner

    # bottom
    for i in range(1, 10):
        xb2 = (s - t_right) - (i - 1) * cell_x
        xb1 = (s - t_right) - i * cell_x
        fields.append(rect(xb1, s - t_bottom, xb2, s))
    fields.append(rect(0, s - t_bottom, t_left, s))  # corner

    # left
    for i in range(1, 10):
        yl2 = (s - t_bottom) - (i - 1) * cell_y
        yl1 = (s - t_bottom) - i * cell_y
        fields.append(rect(0, yl1, t_left, yl2))

    meta = {
        "t_left": t_left, "t_top": t_top, "t_right": t_right, "t_bottom": t_bottom,
        "cell_x": float(cell_x), "cell_y": float(cell_y),
    }
    return fields, meta

def rect_mask(shape_hw: Tuple[int, int], rect: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape_hw
    x1, y1, x2, y2 = rect
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(m, (x1, y1), (x2, y2), 255, thickness=-1)
    return m

def build_masks(warp_size: int, inner_box: Tuple[int, int, int, int]):
    xl, yt, xr, yb = inner_box
    outer = rect_mask((warp_size, warp_size), (0, 0, warp_size, warp_size))
    inner = rect_mask((warp_size, warp_size), (xl, yt, xr, yb))
    ring = cv2.subtract(outer, inner)
    dice_roi = (xl, yt, xr, yb)
    dice_roi_mask = inner.copy()
    return outer, inner, ring, dice_roi, dice_roi_mask



def load_warped_bg_clean(bg_clean_path: str, warp_size: int) -> np.ndarray:
    bg = cv2.imread(bg_clean_path, cv2.IMREAD_COLOR)
    if bg is None:
        raise FileNotFoundError(f"Could not read bg_clean image: {bg_clean_path}")

    if bg.shape[0] != warp_size or bg.shape[1] != warp_size:
        bg = cv2.resize(bg, (warp_size, warp_size), interpolation=cv2.INTER_AREA)

    return bg


def warp_frame(frame_bgr: np.ndarray, H: np.ndarray, warp_size: int) -> np.ndarray:
    return cv2.warpPerspective(frame_bgr, H, (warp_size, warp_size))


def diff_warped_vs_bg_clean(
    frame_bgr: np.ndarray,
    H: np.ndarray,
    warp_size: int,
    bg_clean_bgr: np.ndarray,
    to_gray: bool = False,
    return_stats: bool = False,
) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
    warped = warp_frame(frame_bgr, H, warp_size)

    if bg_clean_bgr.shape[:2] != warped.shape[:2]:
        raise ValueError(
            f"bg_clean shape {bg_clean_bgr.shape[:2]} does not match warped shape {warped.shape[:2]}"
        )
    if bg_clean_bgr.dtype != warped.dtype:
        bg = bg_clean_bgr.astype(warped.dtype, copy=False)
    else:
        bg = bg_clean_bgr

    # Absolute difference
    diff = cv2.absdiff(warped, bg)

    stats = None
    if to_gray or return_stats:
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        if return_stats:
            stats = {
                "mean_diff_gray": float(diff_gray.mean()),
                "p95_diff_gray": float(np.percentile(diff_gray, 95)),
            }

        if to_gray:
            return diff_gray, stats

    return diff, stats



def compute_clean_background_v2(
    video_path, H, warp_size,
    n_samples=240, stride=2,
    mog2_history=500, mog2_varThreshold=14,
    open_k=3, close_k=9,
    min_visible_count=6,
    keep_best_frac=0.25,          # keep best X% frames (least foreground)
    roi_mask=None,                # optional ROI mask in warped coordinates
    return_debug=False
):
    """
    Build a clean 'background' image of the warped board by:
      1) sampling warped frames from a video
      2) using MOG2 to mark foreground (moving objects) per frame
      3) keeping only the cleanest frames (fewest foreground pixels)
      4) computing a per-pixel median over background-only pixels
      5) inpainting pixels that were rarely visible as background

    Args:
        video_path: path to input video
        H: homography for warp_board()
        warp_size: (W,H) size of the warped output
        n_samples: number of warped frames to collect
        stride: take every 'stride'-th frame
        mog2_history / mog2_varThreshold: MOG2 params (bg model stability / sensitivity)
        open_k / close_k: morphology kernel sizes to clean the FG mask
        min_visible_count: minimum number of selected frames where a pixel must be background
        keep_best_frac: fraction of frames to keep (lowest FG ratio)
        roi_mask: optional mask restricting FG ratio scoring to some region
        return_debug: if True, return extra diagnostic values

    Returns:
        bg (H,W,3) uint8 clean background image.
        If return_debug=True: (bg, ratios, keep_idx, holes)
    """
    cap = cv2.VideoCapture(video_path)

    # MOG2 learns a background model; pixels that deviate are flagged as foreground.
    mog = cv2.createBackgroundSubtractorMOG2(
        history=mog2_history,
        varThreshold=mog2_varThreshold,
        detectShadows=False
    )

    frames = []   # collected warped frames
    fgs = []      # corresponding foreground masks (0/255)
    ratios = []   # foreground ratio per sampled frame (cleanliness score)

    i = 0
    # Keep reading until we collect n_samples warped frames (or the video ends)
    while len(frames) < n_samples:
        ok, frame = cap.read()
        if not ok:
            break

        # Sample every `stride` frames to reduce compute and correlation
        if i % stride == 0:
            # Warp into board coordinates so all frames align pixel-to-pixel
            warped = warp_board(frame, H, warp_size=warp_size)

            # Foreground estimation: non-zero pixels indicate "moving objects"
            fg = mog.apply(warped)

            # Binarize to 0/255
            fg = (fg > 0).astype(np.uint8) * 255

            # Morphological OPEN: remove small noise blobs
            if open_k > 1:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
                fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)

            # Morphological CLOSE: fill small holes inside foreground blobs
            if close_k > 1:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
                fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=1)

            # Compute how "clean" this frame is:
            # ratio = (#foreground pixels) / (#pixels considered)
            if roi_mask is not None:
                # Limit scoring to ROI (useful if only inner board matters)
                fg_roi = cv2.bitwise_and(fg, fg, mask=roi_mask)
                # Normalize by ROI area (count of non-zero in roi_mask)
                r = float(np.count_nonzero(fg_roi)) / float(np.count_nonzero(roi_mask))
            else:
                # Score over the whole warped image
                r = float(np.count_nonzero(fg)) / float(fg.size)

            frames.append(warped)
            fgs.append(fg)
            ratios.append(r)

        i += 1

    cap.release()

    # Need enough frames for a reliable median background
    if len(frames) < 20:
        raise RuntimeError("Too few frames for background building v2.")

    # Stack into arrays:
    # frames: (T,H,W,3), fgs: (T,H,W), ratios: (T,)
    frames = np.stack(frames, axis=0).astype(np.uint8)
    fgs = np.stack(fgs, axis=0).astype(np.uint8)
    ratios = np.array(ratios, dtype=float)

    # Keep only the cleanest frames (lowest foreground ratios)
    k = max(10, int(len(ratios) * keep_best_frac))
    keep_idx = np.argsort(ratios)[:k]

    frames_k = frames[keep_idx]   # (k,H,W,3)
    fgs_k = fgs[keep_idx]         # (k,H,W)

    # mask_bg tells, per frame, which pixels are background (True) vs foreground (False)
    mask_bg = (fgs_k == 0)

    # Count, per pixel, how many selected frames consider it background
    visible_count = mask_bg.sum(axis=0)  # (H,W) counts

    # Build background by per-pixel median, but ONLY using background-labeled samples.
    bg = np.zeros_like(frames_k[0], dtype=np.uint8)
    for c in range(3):
        vals = frames_k[..., c].astype(np.float32)  # (k,H,W)

        # Set foreground pixels to NaN so they don't affect the median
        vals[~mask_bg] = np.nan

        # Median across time dimension -> robust background estimate
        med = np.nanmedian(vals, axis=0)  # (H,W)

        # Convert NaNs (where all frames were FG) to 0 for now (will inpaint later)
        bg[..., c] = np.nan_to_num(med, nan=0).astype(np.uint8)

    # Identify "holes": pixels rarely seen as background among selected frames
    holes = (visible_count < min_visible_count).astype(np.uint8) * 255

    # Fill holes using inpainting (propagate nearby texture/color)
    if np.count_nonzero(holes) > 0:
        bg = cv2.inpaint(bg, holes, 3, cv2.INPAINT_TELEA)

    if return_debug:
        return bg, ratios, keep_idx, holes

    return bg
