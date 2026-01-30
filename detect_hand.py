import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


@dataclass
class Params:
    resize_width: int = 960            
    blur_ksize: int = 9                
    bg_alpha: float = 0.03             
    diff_thresh: int = 25              
    motion_open_ksize: int = 5        
    skin_open_ksize: int = 5           
    min_hand_ratio: float = 0.006      
    min_interval_s: float = 0.25       
    merge_gap_s: float = 0.20        
    warmup_frames: int = 30  


def parse_roi(roi_str: str) -> Optional[Tuple[int, int, int, int]]:
    if not roi_str or roi_str.lower() == "none":
        return None
    parts = [int(p.strip()) for p in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be 'x,y,w,h' or 'none'")
    return tuple(parts)  


def resize_keep_aspect(frame: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / float(w)
    new_h = int(round(h * scale))
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def get_skin_mask_bgr(frame_bgr: np.ndarray, open_ksize: int) -> np.ndarray:

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 20, 40], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


def get_motion_mask(gray_blur: np.ndarray, bg_gray: np.ndarray, diff_thresh: int, open_ksize: int) -> np.ndarray:
    diff = cv2.absdiff(gray_blur, bg_gray)
    _, motion = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, k, iterations=1)
    motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, k, iterations=1)
    return motion


def decide_hand_present(hand_mask: np.ndarray, roi: Optional[Tuple[int, int, int, int]], min_ratio: float) -> bool:
    h, w = hand_mask.shape[:2]
    if roi is None:
        x, y, rw, rh = 0, 0, w, h
    else:
        x, y, rw, rh = roi

    x2 = max(0, min(w, x + rw))
    y2 = max(0, min(h, y + rh))
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))

    crop = hand_mask[y:y2, x:x2]
    if crop.size == 0:
        return False

    ratio = float(np.count_nonzero(crop)) / float(crop.size)
    return ratio >= min_ratio


def booleans_to_intervals(flags: List[bool], fps: float) -> List[Tuple[float, float]]:
    intervals = []
    start = None
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            end = i
            intervals.append((start / fps, end / fps))
            start = None
    if start is not None:
        intervals.append((start / fps, len(flags) / fps))
    return intervals


def merge_and_filter_intervals(intervals: List[Tuple[float, float]], min_len: float, merge_gap: float) -> List[Tuple[float, float]]:
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])

    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s - pe <= merge_gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    merged = [(s, e) for (s, e) in merged if (e - s) >= min_len]
    return merged


def invert_intervals(total_s: float, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return [(0.0, total_s)]

    intervals = sorted(intervals, key=lambda x: x[0])
    out = []
    cur = 0.0
    for s, e in intervals:
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < total_s:
        out.append((cur, total_s))
    return out


def process_video(video_path: str, roi: Optional[Tuple[int, int, int, int]], p: Params, debug: bool = False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    hand_flags: List[bool] = []
    bg_gray = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_keep_aspect(frame, p.resize_width)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (p.blur_ksize, p.blur_ksize), 0)

        if bg_gray is None:
            bg_gray = gray_blur.copy().astype(np.float32)

        bg_u8 = cv2.convertScaleAbs(bg_gray)

        skin = get_skin_mask_bgr(frame, p.skin_open_ksize)
        motion = get_motion_mask(gray_blur, bg_u8, p.diff_thresh, p.motion_open_ksize)

        hand_mask = cv2.bitwise_and(skin, motion)

        hand_present = decide_hand_present(hand_mask, roi, p.min_hand_ratio)
        hand_flags.append(hand_present)

        # Update background:
        # - during warmup: always update (to learn the board)
        # - after warmup: update only when we believe there's NO hand
        if frame_idx < p.warmup_frames or (not hand_present):
            cv2.accumulateWeighted(gray_blur, bg_gray, p.bg_alpha)

        if debug and frame_idx % int(max(1, fps // 2)) == 0:
            print(f"[debug] frame={frame_idx}, hand_present={hand_present}")

        frame_idx += 1

    cap.release()

    total_s = len(hand_flags) / fps
    hand_intervals = booleans_to_intervals(hand_flags, fps)
    hand_intervals = merge_and_filter_intervals(hand_intervals, p.min_interval_s, p.merge_gap_s)

    no_hand_intervals = invert_intervals(total_s, hand_intervals)
    no_hand_intervals = merge_and_filter_intervals(no_hand_intervals, p.min_interval_s, p.merge_gap_s)

    return {
        "video": video_path,
        "fps": fps,
        "total_seconds": total_s,
        "roi": None if roi is None else {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]},
        "params": p.__dict__,
        "hand_present_intervals": hand_intervals,
        "no_hand_intervals": no_hand_intervals,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--out", default="hand_periods.json", help="Output JSON path")
    ap.add_argument("--roi", default="none", help="ROI as 'x,y,w,h' or 'none' for full frame")
    ap.add_argument("--debug", action="store_true", help="Print debug lines while processing")

    ap.add_argument("--min_hand_ratio", type=float, default=0.006)
    ap.add_argument("--diff_thresh", type=int, default=25)
    ap.add_argument("--bg_alpha", type=float, default=0.03)
    ap.add_argument("--resize_width", type=int, default=960)

    args = ap.parse_args()

    p = Params(
        min_hand_ratio=args.min_hand_ratio,
        diff_thresh=args.diff_thresh,
        bg_alpha=args.bg_alpha,
        resize_width=args.resize_width,
    )

    roi = parse_roi(args.roi)

    result = process_video(args.video, roi, p, debug=args.debug)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {args.out}")
    print("Hand present intervals:")
    for s, e in result["hand_present_intervals"]:
        print(f"  {s:.2f} - {e:.2f} s")
    print("No-hand intervals (use these for stable processing):")
    for s, e in result["no_hand_intervals"]:
        print(f"  {s:.2f} - {e:.2f} s")


if __name__ == "__main__":
    main()
