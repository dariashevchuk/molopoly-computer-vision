from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np

Box = Tuple[int, int, int, int] 
Quad = np.ndarray 



@dataclass
class CardDetectCFG:
    blur_ksize: int = 5  
    canny1: int = 60
    canny2: int = 160
    # Morphology on edges
    dilate_iter: int = 1
    erode_iter: int = 0

    # Contour/shape filtering
    min_area: int = 2_000
    max_area: int = 200_000
    min_aspect: float = 0.55
    max_aspect: float = 1.80
    approx_eps_frac: float = 0.02  

    use_hsv_filter: bool = True
    h_min: int = 0
    h_max: int = 179
    s_min: int = 0
    s_max: int = 255
    v_min: int = 0
    v_max: int = 255
    hsv_keep_ratio_min: float = 0.15  #

    return_warps: bool = False
    warp_w: int = 200
    warp_h: int = 300



@dataclass
class CardDetections:
    boxes: List[Box]
    quads: List[Quad]
    debug: Dict[str, Any]

    warps: Optional[List[np.ndarray]] = None

    
    crop_offset_xy: Optional[Tuple[int, int]] = None
    boxes_warped: Optional[List[Box]] = None
    quads_warped: Optional[List[Quad]] = None

    def draw_on_image(
        self,
        image_bgr: np.ndarray,
        *,
        use_warped_coords: bool = False,
        color_quad: Tuple[int, int, int] = (0, 255, 0),
        color_box: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        out = image_bgr.copy()
        if use_warped_coords:
            if self.quads_warped is None or self.boxes_warped is None:
                raise ValueError("no warped coords available in this CardDetections")
            quads = self.quads_warped
            boxes = self.boxes_warped
        else:
            quads = self.quads
            boxes = self.boxes

        if len(quads):
            cv2.drawContours(out, quads, -1, color_quad, thickness)
        for (x, y, w, h) in boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), color_box, thickness)
        return out

    def draw_on_warped(
        self,
        warped_bgr: np.ndarray,
        *,
        inner_box: Optional[Tuple[int, int, int, int]] = None,
        draw_inner_box: bool = True,
        inner_box_color: Tuple[int, int, int] = (0, 255, 255),
        **kwargs,
    ) -> np.ndarray:

        out = warped_bgr.copy()
        if draw_inner_box and inner_box is not None:
            xl, yt, xr, yb = inner_box
            cv2.rectangle(out, (int(xl), int(yt)), (int(xr), int(yb)), inner_box_color, 2)
        if self.quads_warped is None or self.boxes_warped is None:
            return self.draw_on_image(out, use_warped_coords=False, **kwargs)
        return self.draw_on_image(out, use_warped_coords=True, **kwargs)


def _odd_ksize(k: int) -> int:
    k = int(k)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    return k


def _order_points(pts: np.ndarray) -> np.ndarray:
    # top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_cards_in_crop(
    crop_bgr: np.ndarray,
    cfg: Optional[CardDetectCFG] = None,
) -> CardDetections:

    if cfg is None:
        cfg = CardDetectCFG()

    img = crop_bgr.copy()

    k = _odd_ksize(cfg.blur_ksize)
    blurred = cv2.GaussianBlur(img, (k, k), 0)

    edges = cv2.Canny(blurred, int(cfg.canny1), int(cfg.canny2))

    morph = edges.copy()
    if cfg.dilate_iter > 0:
        morph = cv2.dilate(morph, None, iterations=int(cfg.dilate_iter))
    if cfg.erode_iter > 0:
        morph = cv2.erode(morph, None, iterations=int(cfg.erode_iter))

    cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_mask = None
    if cfg.use_hsv_filter:
        lower = np.array([cfg.h_min, cfg.s_min, cfg.v_min], dtype=np.uint8)
        upper = np.array([cfg.h_max, cfg.s_max, cfg.v_max], dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv, lower, upper)

    boxes: List[Box] = []
    quads: List[Quad] = []
    warps: List[np.ndarray] = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < cfg.min_area or area > cfg.max_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, float(cfg.approx_eps_frac) * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if w <= 0 or h <= 0:
            continue
        aspect = w / float(h)
        if not (cfg.min_aspect <= aspect <= cfg.max_aspect):
            continue

        if cfg.use_hsv_filter and hsv_mask is not None:
            quad_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(quad_mask, approx.reshape(-1, 2), 255)
            inside = cv2.bitwise_and(hsv_mask, hsv_mask, mask=quad_mask)
            keep_ratio = (inside > 0).sum() / max(1, (quad_mask > 0).sum())
            if keep_ratio < cfg.hsv_keep_ratio_min:
                continue

        boxes.append((int(x), int(y), int(w), int(h)))
        quads.append(approx)

        if cfg.return_warps:
            pts = approx.reshape(4, 2).astype("float32")
            rect = _order_points(pts)
            dst = np.array(
                [
                    [0, 0],
                    [cfg.warp_w - 1, 0],
                    [cfg.warp_w - 1, cfg.warp_h - 1],
                    [0, cfg.warp_h - 1],
                ],
                dtype="float32",
            )
            M = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(img, M, (cfg.warp_w, cfg.warp_h))
            warps.append(warp)

    overlay = img.copy()
    if len(quads):
        cv2.drawContours(overlay, quads, -1, (0, 255, 0), 2)
    for (x, y, w, h) in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)

    debug: Dict[str, Any] = {
        "blurred": blurred,
        "edges": edges,
        "morph": morph,
        "hsv_mask": hsv_mask,
        "overlay": overlay,
    }

    return CardDetections(
        boxes=boxes,
        quads=quads,
        debug=debug,
        warps=warps if cfg.return_warps else None,
        crop_offset_xy=None,
        boxes_warped=None,
        quads_warped=None,
    )



def _map_box_offset(box: Box, dx: int, dy: int) -> Box:
    x, y, w, h = box
    return (x + dx, y + dy, w, h)


def _map_quad_offset(quad: Quad, dx: int, dy: int) -> Quad:
    q = quad.copy()
    q[:, 0, 0] = q[:, 0, 0] + dx
    q[:, 0, 1] = q[:, 0, 1] + dy
    return q


def crop_inner_field(
    warped_bgr: np.ndarray,
    inner_box: Tuple[int, int, int, int],
) -> np.ndarray:
    """Return the inner-field crop from a warped board."""
    xl, yt, xr, yb = map(int, inner_box)
    return warped_bgr[yt:yb, xl:xr].copy()


def detect_cards_in_inner_field(
    warped_bgr: np.ndarray,
    inner_box: Tuple[int, int, int, int],
    cfg: Optional[CardDetectCFG] = None,
) -> CardDetections:

    if cfg is None:
        cfg = CardDetectCFG()

    xl, yt, xr, yb = map(int, inner_box)
    if xl < 0 or yt < 0 or xr <= xl or yb <= yt:
        raise ValueError(f"Invalid inner_box: {inner_box}")

    crop = warped_bgr[yt:yb, xl:xr].copy()
    det = detect_cards_in_crop(crop, cfg=cfg)

    det.crop_offset_xy = (xl, yt)
    det.boxes_warped = [_map_box_offset(b, xl, yt) for b in det.boxes]
    det.quads_warped = [_map_quad_offset(q, xl, yt) for q in det.quads]
    return det


def map_points_warped_to_frame(
    pts_xy: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    
    if pts_xy.ndim != 2 or pts_xy.shape[1] != 2:
        raise ValueError("pts_xy must have shape (N,2)")
    Hinv = np.linalg.inv(H)
    pts = pts_xy.astype(np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, Hinv)
    return mapped.reshape(-1, 2)


def map_quad_warped_to_frame(
    quad_warped: Quad,
    H: np.ndarray,
) -> np.ndarray:
    pts = quad_warped.reshape(-1, 2).astype(np.float32)
    return map_points_warped_to_frame(pts, H)
