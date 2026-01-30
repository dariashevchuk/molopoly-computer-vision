# detect_dice.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  

@dataclass
class TopFaceCropCFG:
    out_size: int = 650
    upscale_long_side: int = 650
    ab_thr: int = 35
    L_min: int = 175
    L_quantile: int = 94
    open_k: int = 7
    close_k: int = 15
    close_iter: int = 1
    pad_frac: float = 0.03
    debug: bool = False


@dataclass
class PipCountCFG:
    ab_thr: int = 55
    L_lo: int = 165
    open_k: int = 3
    close_k: int = 9
    close_iter: int = 1
    border_frac: float = 0.045
    min_area_frac: float = 0.0012
    max_area_frac: float = 0.06
    max_aspect: float = 1.6
    keep_k: int = 6
    neighbor_dist_frac: float = 0.20
    grid_snap_frac: float = 0.10
    debug: bool = False


@dataclass
class DiceRegionCFG:
    # Dice region detection inside inner ROI (neutral + bright)
    ab_thr: int = 38
    L_min: int = 170
    L_quantile: int = 92

    # Morphology (smaller defaults help tiny dice survive)
    open_k: int = 5
    close_k: int = 13
    close_iter: int = 1


    min_area_frac: float = 0.00025
    max_area_frac: float = 0.06

    max_aspect: float = 1.35

    use_rotated_rect: bool = True
    max_rot_aspect: float = 1.25       
    min_extent: float = 0.55          
    require_quad_like: bool = False   
    quad_eps_frac: float = 0.04       

    # Selection
    keep_top_k: int = 2
    pad_frac: float = 0.06
    max_candidates_to_score: int = 12   

    # Run pips pipeline on each detected die crop and use it to score candidates
    run_pips: bool = True
    score_with_pips: bool = True        # prefer candidates that yield 1..6 pips

    debug_keep_color_roi: bool = True


@dataclass
class DiceDetections:
    boxes: List[Box]                          
    boxes_warped: Optional[List[Box]]         
    dice_values: Optional[List[int]]        
    crops: Optional[List[np.ndarray]]     
    debug: Dict[str, Any]



def _odd(k: int) -> int:
    k = int(k)
    if k < 1:
        return 1
    return k if (k % 2 == 1) else (k + 1)


def _order_quad(pts4: np.ndarray) -> np.ndarray:
    pts = np.array(pts4, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl], axis=0)


def _clamp_box(x: int, y: int, w: int, h: int, W: int, H: int) -> Box:
    x1 = max(0, min(W - 1, x))
    y1 = max(0, min(H - 1, y))
    x2 = max(x1 + 1, min(W, x + w))
    y2 = max(y1 + 1, min(H, y + h))
    return (x1, y1, x2 - x1, y2 - y1)


def _pad_box(box: Box, W: int, H: int, pad_frac: float) -> Box:
    x, y, w, h = box
    pad = int(round(pad_frac * max(w, h)))
    return _clamp_box(x - pad, y - pad, w + 2 * pad, h + 2 * pad, W, H)


def _quantize_1d(vals: List[float], tol: float) -> List[float]:
    groups: List[List[float]] = []
    for v in sorted(vals):
        placed = False
        for g in groups:
            if abs(v - g[0]) <= tol:
                g.append(v)
                g[0] = float(np.mean(g[1:]))
                placed = True
                break
        if not placed:
            groups.append([float(v), float(v)])
    return [g[0] for g in groups]


def _contour_square_features(c: np.ndarray, quad_eps_frac: float = 0.04) -> Dict[str, Any]:
    area = float(cv2.contourArea(c))
    rect = cv2.minAreaRect(c) 
    (wr, hr) = rect[1]
    wr = float(wr)
    hr = float(hr)
    rot_aspect = 999.0
    rect_area = (wr * hr) if (wr > 1e-6 and hr > 1e-6) else 0.0
    if rect_area > 0:
        rot_aspect = max(wr / hr, hr / wr)
    extent = (area / rect_area) if rect_area > 0 else 0.0

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, float(quad_eps_frac) * peri, True) if peri > 0 else c
    quad_like = (len(approx) == 4)

    return {
        "area": area,
        "rot_aspect": float(rot_aspect),
        "extent": float(extent),
        "quad_like": bool(quad_like),
        "rect": rect,
    }



def _mask_neutral_bright_lab(bgr: np.ndarray, ab_thr: int, L_min: int, L_quantile: int) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, bb = cv2.split(lab)

    neutral = ((np.abs(a.astype(np.int16) - 128) <= int(ab_thr)) &
               (np.abs(bb.astype(np.int16) - 128) <= int(ab_thr)))

    if np.any(neutral):
        thr = max(int(L_min), int(np.percentile(L[neutral], int(L_quantile))))
    else:
        thr = int(L_min)

    bright = (L >= thr)
    mask = (neutral & bright).astype(np.uint8) * 255
    return mask



def crop_die_top_face_square_with_border(
    die_bgr: np.ndarray,
    cfg: Optional[TopFaceCropCFG] = None,
) -> np.ndarray:

    if cfg is None:
        cfg = TopFaceCropCFG()

    bgr0 = die_bgr.copy()

    h0, w0 = bgr0.shape[:2]
    s = float(cfg.upscale_long_side) / float(max(h0, w0))
    if abs(s - 1.0) < 1e-6:
        bgr = bgr0.copy()
    else:
        bgr = cv2.resize(
            bgr0,
            (max(1, int(w0 * s)), max(1, int(h0 * s))),
            interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
        )

    mask = _mask_neutral_bright_lab(bgr, cfg.ab_thr, cfg.L_min, cfg.L_quantile)

    ok = int(cfg.open_k) | 1
    ck = int(cfg.close_k) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok)),
                            iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)),
                            iterations=int(cfg.close_iter))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        H, W = bgr.shape[:2]
        side = min(H, W)
        y0 = (H - side) // 2
        x0 = (W - side) // 2
        face_sq = cv2.resize(bgr[y0:y0 + side, x0:x0 + side],
                             (cfg.out_size, cfg.out_size),
                             interpolation=cv2.INTER_AREA)
        return face_sq

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 80:  
        face_sq = cv2.resize(bgr, (cfg.out_size, cfg.out_size), interpolation=cv2.INTER_AREA)
        return face_sq

    peri = cv2.arcLength(cnt, True)
    quad = None
    for eps_frac in (0.02, 0.03, 0.04, 0.05, 0.06):
        approx = cv2.approxPolyDP(cnt, eps_frac * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        rect = cv2.minAreaRect(cnt)
        quad = cv2.boxPoints(rect).astype(np.float32)

    src = _order_quad(quad)

    c = src.mean(axis=0, keepdims=True)
    src = (src - c) * (1.0 + float(cfg.pad_frac)) + c

    dst = np.array([[0, 0],
                    [cfg.out_size - 1, 0],
                    [cfg.out_size - 1, cfg.out_size - 1],
                    [0, cfg.out_size - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    face_sq = cv2.warpPerspective(
        bgr, M, (cfg.out_size, cfg.out_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return face_sq



def count_pips_on_cropped_top_face(
    face_bgr: np.ndarray,
    cfg: Optional[PipCountCFG] = None
) -> Dict[str, Any]:
    
    if cfg is None:
        cfg = PipCountCFG()

    bgr = face_bgr.copy()
    h, w = bgr.shape[:2]

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, bb = cv2.split(lab)

    neutral = ((np.abs(a.astype(np.int16) - 128) <= int(cfg.ab_thr)) &
               (np.abs(bb.astype(np.int16) - 128) <= int(cfg.ab_thr))).astype(np.uint8) * 255
    bright_enough = (L >= int(cfg.L_lo)).astype(np.uint8) * 255

    face_mask = cv2.bitwise_and(neutral, bright_enough)

    ok = int(cfg.open_k) | 1
    ck = int(cfg.close_k) | 1
    face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok)),
                                 iterations=1)
    face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)),
                                 iterations=int(cfg.close_iter))

    pips = cv2.bitwise_not(face_mask)

    bx = int(cfg.border_frac * w)
    by = int(cfg.border_frac * h)
    pips[:by, :] = 0
    pips[-by:, :] = 0
    pips[:, :bx] = 0
    pips[:, -bx:] = 0

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(pips, connectivity=8)

    area_img = h * w
    min_area = int(cfg.min_area_frac * area_img)
    max_area = int(cfg.max_area_frac * area_img)

    centers: List[Tuple[float, float]] = []
    areas: List[int] = []

    for i in range(1, num):
        x, y, ww, hh, aarea = stats[i]
        if aarea < min_area or aarea > max_area:
            continue
        asp = max(ww / max(1, hh), hh / max(1, ww))
        if asp > cfg.max_aspect:
            continue
        cx, cy = centroids[i]
        centers.append((float(cx), float(cy)))
        areas.append(int(aarea))

    raw_count = len(centers)

    if raw_count <= 6:
        kept = list(range(raw_count))
    else:
        P = np.array(centers, dtype=np.float32)
        D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2))

        dthr = cfg.neighbor_dist_frac * float(min(h, w))
        neigh = (D <= dthr).astype(np.int32)
        np.fill_diagonal(neigh, 0)
        deg = neigh.sum(axis=1)

        seed = int(np.argmax(deg))
        cand = np.where(D[seed] <= dthr)[0].tolist() + [seed]
        cand = sorted(set(cand))

        if len(cand) > cfg.keep_k:
            sub = cand.copy()
            kept = [seed]
            sub.remove(seed)
            while len(kept) < cfg.keep_k and sub:
                best_j = None
                best_score = -1e18
                for j in sub:
                    score = -float(np.mean([D[j, k] for k in kept]))
                    if score > best_score:
                        best_score = score
                        best_j = j
                kept.append(best_j)
                sub.remove(best_j)
        else:
            kept = cand

        kept_pts = P[kept]
        x_tol = cfg.grid_snap_frac * w
        y_tol = cfg.grid_snap_frac * h
        xs = _quantize_1d(kept_pts[:, 0].tolist(), x_tol)
        ys = _quantize_1d(kept_pts[:, 1].tolist(), y_tol)

        if len(xs) <= 3 and len(ys) > 3:
            ys_sorted = sorted(ys)
            low = ys_sorted[-1]
            new_kept = [k for k in kept if abs(P[k, 1] - low) > y_tol]
            if 1 <= len(new_kept) <= 6:
                kept = new_kept

    pip_count = len(kept)
    kept_centers = [centers[i] for i in kept] if raw_count else []
    kept_areas = [areas[i] for i in kept] if raw_count else []

    return {
        "pip_count": int(pip_count),
        "raw_count": int(raw_count),
        "centers": kept_centers,
        "areas": kept_areas,
        "face_mask": face_mask,
        "pips_mask": pips,
    }


def read_top_face_pips(
    die_bgr: np.ndarray,
    crop_cfg: Optional[TopFaceCropCFG] = None,
    pip_cfg: Optional[PipCountCFG] = None,
) -> Dict[str, Any]:

    face = crop_die_top_face_square_with_border(die_bgr, cfg=crop_cfg)
    res = count_pips_on_cropped_top_face(face, cfg=pip_cfg)
    res["face"] = face
    return res


def detect_dice_in_crop(
    crop_bgr: np.ndarray,
    region_cfg: Optional[DiceRegionCFG] = None,
    crop_cfg: Optional[TopFaceCropCFG] = None,
    pip_cfg: Optional[PipCountCFG] = None,
) -> DiceDetections:

    if region_cfg is None:
        region_cfg = DiceRegionCFG()
    if crop_cfg is None:
        crop_cfg = TopFaceCropCFG(debug=False)
    if pip_cfg is None:
        pip_cfg = PipCountCFG(debug=False)

    img = crop_bgr.copy()
    H, W = img.shape[:2]
    roi_area = float(H * W)

    raw_mask = _mask_neutral_bright_lab(img, region_cfg.ab_thr, region_cfg.L_min, region_cfg.L_quantile)

    ok = _odd(region_cfg.open_k)
    ck = _odd(region_cfg.close_k)
    mask = cv2.morphologyEx(
        raw_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok)),
        iterations=1
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)),
        iterations=int(region_cfg.close_iter)
    )

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = region_cfg.min_area_frac * roi_area
    max_area = region_cfg.max_area_frac * roi_area

    # Build candidate list
 
    candidates: List[Dict[str, Any]] = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue
        aspect = max(w / float(h), h / float(w))
        if aspect > float(region_cfg.max_aspect):
            continue

        feats = _contour_square_features(c, quad_eps_frac=float(region_cfg.quad_eps_frac))
        if region_cfg.use_rotated_rect:
            if feats["rot_aspect"] > float(region_cfg.max_rot_aspect):
                continue
            if feats["extent"] < float(region_cfg.min_extent):
                continue
        if region_cfg.require_quad_like and (not feats["quad_like"]):
            continue

        candidates.append({
            "area": area,
            "box": (int(x), int(y), int(w), int(h)),
            "feats": feats,
            "score": 0.0,
        })

    # Early sort by area 
    candidates.sort(key=lambda d: float(d["area"]), reverse=True)

    scored_debug: List[Dict[str, Any]] = []
    if region_cfg.run_pips and region_cfg.score_with_pips and candidates:
        for d in candidates[: int(region_cfg.max_candidates_to_score)]:
            b = d["box"]
            pb = _pad_box(b, W, H, region_cfg.pad_frac)
            x, y, w, h = pb
            die_crop = img[y:y + h, x:x + w].copy()

            r = read_top_face_pips(die_crop, crop_cfg=crop_cfg, pip_cfg=pip_cfg)
            pip = int(r["pip_count"])
            raw = int(r["raw_count"])

            # Score heuristic:
            # - big reward if pip is 1..6
            # - smaller reward if raw_count isn't crazy
            # - add slight preference for reasonable area (prevents tiny specks)
            score = 0.0
            if 1 <= pip <= 6:
                score += 100.0
                score += max(0.0, 10.0 - abs(float(raw - pip)) * 2.0)
            else:
                score -= 20.0

            if raw > 20:
                score -= float(raw - 20)

            score += 0.0005 * float(d["area"])

            d["score"] = float(score)
            d["pip"] = pip
            d["raw"] = raw
            scored_debug.append({
                "box": d["box"],
                "area": float(d["area"]),
                "score": float(score),
                "pip_count": pip,
                "raw_count": raw,
                "rot_aspect": float(d["feats"]["rot_aspect"]),
                "extent": float(d["feats"]["extent"]),
                "quad_like": bool(d["feats"]["quad_like"]),
            })

        candidates.sort(key=lambda d: (float(d["score"]), float(d["area"])), reverse=True)

    boxes = [d["box"] for d in candidates[: int(region_cfg.keep_top_k)]]

    boxes = sorted(boxes, key=lambda b: (b[0] + b[2] / 2.0))

    crops: List[np.ndarray] = []
    for b in boxes:
        pb = _pad_box(b, W, H, region_cfg.pad_frac)
        x, y, w, h = pb
        crops.append(img[y:y + h, x:x + w].copy())

    dice_values: Optional[List[int]] = None
    pip_debug: List[Dict[str, Any]] = []

    
    if region_cfg.run_pips:
        dice_values = []
        if region_cfg.score_with_pips and "scored_debug" in locals():
            br_map = {(d["box"][0], d["box"][1], d["box"][2], d["box"][3]): (int(d.get("pip", 0)), int(d.get("raw", 0)))
                      for d in candidates}
            for die_crop, b in zip(crops, boxes):
                pip, raw = br_map.get((b[0], b[1], b[2], b[3]), (0, 0))
                if not (1 <= pip <= 6):
                    r = read_top_face_pips(die_crop, crop_cfg=crop_cfg, pip_cfg=pip_cfg)
                    pip = int(r["pip_count"])
                    raw = int(r["raw_count"])
                dice_values.append(int(pip))
                pip_debug.append({"pip_count": int(pip), "raw_count": int(raw)})
        else:
            for die_crop in crops:
                r = read_top_face_pips(die_crop, crop_cfg=crop_cfg, pip_cfg=pip_cfg)
                dice_values.append(int(r["pip_count"]))
                pip_debug.append({
                    "pip_count": int(r["pip_count"]),
                    "raw_count": int(r["raw_count"]),
                })

    overlay = img.copy()
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        lab = f"dice {i}"
        if dice_values is not None and i - 1 < len(dice_values):
            lab += f"={dice_values[i-1]}"
        cv2.putText(
            overlay, lab, (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
        )

    dbg: Dict[str, Any] = {
        "mask_raw": raw_mask,
        "mask_morph": mask,
        "overlay": overlay,
        "num_contours": int(len(cnts)),
        "num_candidates": int(len(candidates)),
        "pip_debug": pip_debug,
    }
    if "scored_debug" in locals():
        dbg["scored_candidates"] = scored_debug
    if getattr(region_cfg, "debug_keep_color_roi", False):
        dbg["roi_bgr"] = img

    return DiceDetections(
        boxes=boxes,
        boxes_warped=None,
        dice_values=dice_values,
        crops=crops if crops else None,
        debug=dbg,
    )


def detect_dice_in_inner_field(
    warped_bgr: np.ndarray,
    inner_box: Tuple[int, int, int, int],
    region_cfg: Optional[DiceRegionCFG] = None,
    crop_cfg: Optional[TopFaceCropCFG] = None,
    pip_cfg: Optional[PipCountCFG] = None,
) -> DiceDetections:

    xl, yt, xr, yb = map(int, inner_box)
    if xl < 0 or yt < 0 or xr <= xl or yb <= yt:
        raise ValueError(f"Invalid inner_box: {inner_box}")

    crop = warped_bgr[yt:yb, xl:xr].copy()
    det = detect_dice_in_crop(crop, region_cfg=region_cfg, crop_cfg=crop_cfg, pip_cfg=pip_cfg)
    det.boxes_warped = [(x + xl, y + yt, w, h) for (x, y, w, h) in det.boxes]
    if det.debug is not None and (region_cfg is None or getattr(region_cfg, "debug_keep_color_roi", False)):
        det.debug["inner_roi_bgr"] = crop
        det.debug["inner_box_warped"] = (xl, yt, xr, yb)
    return det


def draw_dice_on_warped(
    warped_bgr: np.ndarray,
    det: DiceDetections,
    inner_box: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:

    out = warped_bgr.copy()

    if inner_box is not None:
        xl, yt, xr, yb = map(int, inner_box)
        cv2.rectangle(out, (xl, yt), (xr, yb), (0, 255, 255), 2)

    boxes = det.boxes_warped if det.boxes_warped is not None else det.boxes
    boxes = sorted(list(boxes), key=lambda b: (b[0] + b[2] / 2.0))

    for i, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        lab = f"dice {i}"
        if det.dice_values is not None and (i - 1) < len(det.dice_values):
            lab += f"={int(det.dice_values[i-1])}"
        cv2.putText(
            out, lab, (int(x), max(0, int(y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
        )

    return out
