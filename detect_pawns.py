from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union

import cv2
import numpy as np

try:
    from detect_board import warp_board, load_warped_bg_clean, compute_clean_background_v2
except Exception as e:
    warp_board = None
    load_warped_bg_clean = None
    compute_clean_background_v2 = None



@dataclass
class PawnCFG:
    thr_q: float = 99.0          # percentile on diff values inside ROI
    mad_k: float = 4.5           # threshold = med + mad_k * MAD
    thr_min: int = 5
    thr_max: int = 255

    open_k: int = 3
    close_k: int = 7
    open_iter: int = 1
    close_iter: int = 2

    topk_contours: int = 12

    min_area_ratio: float = 0.00008
    max_area_ratio: float = 0.08

    min_extent: float = 0.10
    min_solidity: float = 0.20
    min_ar: float = 0.25
    max_ar: float = 4.0

    # Optional 
    min_area_px: int = 0
    max_area_px: int = 0



@dataclass
class PawnBGCFG:

    mode: str = "estimated"  # static or estimated

    # For static mode
    bg_clean_path: str = "warped_board_clean.png"

    # For estimated mode
    video_path: Optional[str] = None
    n_samples: int = 240
    stride: int = 2
    mog2_history: int = 500
    mog2_varThreshold: int = 14
    open_k: int = 3
    close_k: int = 9
    min_visible_count: int = 6
    keep_best_frac: float = 0.25


class PawnBackgroundProvider:
    def __init__(
        self,
        cfg: PawnBGCFG,
        H: np.ndarray,
        warp_size: int,
        roi_mask: Optional[np.ndarray] = None,
    ):
        self.cfg = cfg
        self.H = H
        self.warp_size = int(warp_size)
        self.roi_mask = roi_mask
        self._bg: Optional[np.ndarray] = None

    def get(self) -> np.ndarray:
        if self._bg is not None:
            return self._bg

        mode = (self.cfg.mode or "").lower().strip()
        if mode not in ("static", "estimated"):
            raise ValueError(f"PawnBGCFG.mode must be 'static' or 'estimated', got: {self.cfg.mode}")

        if mode == "static":
            if load_warped_bg_clean is None:
                raise ImportError("Could not import load_warped_bg_clean from detect_board.py")
            self._bg = load_warped_bg_clean(self.cfg.bg_clean_path, self.warp_size)
            return self._bg

        # estimated
        if compute_clean_background_v2 is None:
            raise ImportError("Could not import compute_clean_background_v2 from detect_board.py")
        if not self.cfg.video_path:
            raise ValueError("PawnBGCFG.video_path is required when mode='estimated'")

        self._bg = compute_clean_background_v2(
            video_path=self.cfg.video_path,
            H=self.H,
            warp_size=self.warp_size,
            n_samples=int(self.cfg.n_samples),
            stride=int(self.cfg.stride),
            mog2_history=int(self.cfg.mog2_history),
            mog2_varThreshold=int(self.cfg.mog2_varThreshold),
            open_k=int(self.cfg.open_k),
            close_k=int(self.cfg.close_k),
            min_visible_count=int(self.cfg.min_visible_count),
            keep_best_frac=float(self.cfg.keep_best_frac),
            roi_mask=self.roi_mask,
            return_debug=False,
        )
        return self._bg



def _binary_cleanup(m: np.ndarray, cfg: PawnCFG) -> np.ndarray:
    m = m.copy()
    if cfg.open_k and cfg.open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_k, cfg.open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=cfg.open_iter)
    if cfg.close_k and cfg.close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_k, cfg.close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=cfg.close_iter)
    return m


def _contour_components(bin_mask: np.ndarray) -> List[Dict[str, Any]]:
    cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Dict[str, Any]] = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area <= 0:
            continue

        x, y, w, h = cv2.boundingRect(c)
        ar = float(w) / float(h + 1e-6)
        extent = area / float(w * h + 1e-6)

        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull)) + 1e-6
        solidity = area / hull_area

        peri = float(cv2.arcLength(c, True)) + 1e-6
        circularity = 4.0 * np.pi * area / (peri * peri)

        cx = x + w / 2.0
        cy = y + h / 2.0

        out.append({
            "area": area,
            "bbox": (int(x), int(y), int(x + w), int(y + h)),
            "center": (float(cx), float(cy)),
            "ar": ar,
            "extent": extent,
            "solidity": solidity,
            "circularity": circularity,
            "contour": c,
        })

    out.sort(key=lambda d: d["area"], reverse=True)
    return out


def _robust_threshold(vals: np.ndarray, q: float, mad_k: float, thr_min: int, thr_max: int) -> int:
    if vals.size == 0:
        return int(np.clip(22, thr_min, thr_max))  

    vals = vals.astype(np.float32, copy=False)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-6
    thr_mad = med + mad_k * mad
    thr_q = float(np.percentile(vals, q))
    thr = max(thr_mad, thr_q)
    return int(np.clip(thr, thr_min, thr_max))


def _diff_warped_vs_bg_clean(frame_bgr: np.ndarray, H: np.ndarray, warp_size: int, bg_clean_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    if warp_board is None:
        raise ImportError("Could not import warp_board from detect_board.py")
    warped = warp_board(frame_bgr, H, warp_size)
    if bg_clean_bgr.shape[:2] != warped.shape[:2]:
        raise ValueError(f"bg_clean shape {bg_clean_bgr.shape[:2]} != warped shape {warped.shape[:2]}")
    if bg_clean_bgr.dtype != warped.dtype:
        bg = bg_clean_bgr.astype(warped.dtype, copy=False)
    else:
        bg = bg_clean_bgr
    diff = cv2.absdiff(warped, bg)
    return warped, diff



def detect_pawns_from_diff(
    diff_frame: np.ndarray,
    ring_mask: np.ndarray,
    prev_centers: Optional[List[Tuple[float, float]]] = None,
    cfg: Optional[PawnCFG] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:

    if cfg is None:
        cfg = PawnCFG()

    if diff_frame.ndim == 3:
        diff_gray = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
    else:
        diff_gray = diff_frame

    roi = (ring_mask > 0)
    vals = diff_gray[roi].reshape(-1)
    thr = _robust_threshold(vals, cfg.thr_q, cfg.mad_k, cfg.thr_min, cfg.thr_max)

    _, fg = cv2.threshold(diff_gray, thr, 255, cv2.THRESH_BINARY)
    fg = cv2.bitwise_and(fg, fg, mask=ring_mask)
    fg = _binary_cleanup(fg, cfg)

    comps = _contour_components(fg)
    ring_area = float(max(1, np.count_nonzero(ring_mask)))

    cand: List[Dict[str, Any]] = []
    for c in comps[: cfg.topk_contours]:
        area = float(c["area"])
        area_ratio = area / ring_area

        if cfg.min_area_px and area < cfg.min_area_px:
            continue
        if cfg.max_area_px and area > cfg.max_area_px:
            continue

        if area_ratio < cfg.min_area_ratio or area_ratio > cfg.max_area_ratio:
            continue
        if c["extent"] < cfg.min_extent:
            continue
        if c["solidity"] < cfg.min_solidity:
            continue
        if c["ar"] < cfg.min_ar or c["ar"] > cfg.max_ar:
            continue

        cand.append(c)

    # pick up to 2
    picked: List[Dict[str, Any]] = []
    if prev_centers is None or len(prev_centers) == 0:
        picked = cand[:2]
    else:
        def dist2(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

        used = set()
        for pc in prev_centers[:2]:
            best_i, best_d = None, 1e18
            for i, c in enumerate(cand):
                if i in used:
                    continue
                d = dist2(pc, c["center"])
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is not None:
                used.add(best_i)
                picked.append(cand[best_i])

        # fill remaining slot by area
        if len(picked) < 2:
            for i in range(len(cand)):
                if i not in used:
                    picked.append(cand[i])
                if len(picked) == 2:
                    break

    pawns = [{"bbox": p["bbox"], "center": p["center"], "area": int(p["area"])} for p in picked[:2]]

    dbg = {
        "thr": int(thr),
        "n_contours": int(len(comps)),
        "n_candidates": int(len(cand)),
        "candidates": [
            {
                "area": float(c["area"]),
                "center": c["center"],
                "bbox": c["bbox"],
                "extent": float(c["extent"]),
                "solidity": float(c["solidity"]),
                "ar": float(c["ar"]),
                "circularity": float(c["circularity"]),
            }
            for c in cand
        ],
    }
    return pawns, fg, dbg



def detect_pawns_from_frame(
    frame_bgr: np.ndarray,
    H: np.ndarray,
    warp_size: int,
    ring_mask: np.ndarray,
    bg: Union[np.ndarray, PawnBackgroundProvider],
    prev_centers: Optional[List[Tuple[float, float]]] = None,
    cfg: Optional[PawnCFG] = None,
    *,
    return_debug_images: bool = False,
) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:

    bg_clean = bg.get() if isinstance(bg, PawnBackgroundProvider) else bg
    warped, diff_bgr = _diff_warped_vs_bg_clean(frame_bgr, H, int(warp_size), bg_clean)

    pawns, fg, dbg = detect_pawns_from_diff(
        diff_bgr,
        ring_mask=ring_mask,
        prev_centers=prev_centers,
        cfg=cfg,
    )

    if return_debug_images:
        dbg = dict(dbg)
        dbg["warped"] = warped
        dbg["diff_bgr"] = diff_bgr

    return pawns, fg, dbg


