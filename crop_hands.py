#!/usr/bin/env python3


import os
import sys
import shutil
import argparse
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── MediaPipe hands ───────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands

# ── Helpers ───────────────────────────────────────────────────────────────────

def expand_box(x1, y1, x2, y2, img_w, img_h, padding: float = 0.15):
    """Add proportional padding around a bounding box and clamp to image."""
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)
    return x1, y1, x2, y2


def detect_and_crop(
    img_bgr: np.ndarray,
    hands_detector,
    padding: float = 0.15,
):
    """
    Detect hand(s) in *img_bgr* and return a cropped BGR image.

    Strategy
    --------
    1. Try MediaPipe with static_image_mode=True (best for photos).
    2. If no hand found, return the original image unchanged (with a flag).
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(img_rgb)

    if not result.multi_hand_landmarks:
        return img_bgr, False  # no hand found

    # Gather all landmark points across all detected hands
    all_x, all_y = [], []
    for hand_lms in result.multi_hand_landmarks:
        for lm in hand_lms.landmark:
            all_x.append(int(lm.x * w))
            all_y.append(int(lm.y * h))

    x1, y1 = min(all_x), min(all_y)
    x2, y2 = max(all_x), max(all_y)
    x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, padding)

    cropped = img_bgr[y1:y2, x1:x2]
    return cropped, True


def process_dataset(
    src_root: Path,
    dst_root: Path,
    padding: float = 0.15,
    copy_on_fail: bool = True,
    min_detection_confidence: float = 0.5,
):
    """Walk *src_root*, crop hands, write mirrored tree under *dst_root*."""

    # Collect all image paths first for a nice progress bar
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    all_images = [
        p
        for p in src_root.rglob("*")
        if p.is_file() and p.suffix.lower() in image_exts
    ]

    if not all_images:
        print(f"[ERROR] No images found under {src_root}")
        sys.exit(1)

    print(f"Found {len(all_images)} images across {src_root}")
    print(f"Output root  → {dst_root}\n")

    stats = {"cropped": 0, "no_hand": 0, "error": 0}

    # Re-use a single detector instance for speed
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
    ) as detector:
        for img_path in tqdm(all_images, unit="img", desc="Processing"):
            # Mirror the relative path under dst_root
            rel = img_path.relative_to(src_root)
            out_path = dst_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError("cv2.imread returned None")

                cropped, found = detect_and_crop(img, detector, padding)

                if found:
                    cv2.imwrite(str(out_path), cropped)
                    stats["cropped"] += 1
                else:
                    if copy_on_fail:
                        shutil.copy2(img_path, out_path)
                    stats["no_hand"] += 1

            except Exception as exc:
                tqdm.write(f"[WARN] {img_path.name}: {exc}")
                if copy_on_fail:
                    try:
                        shutil.copy2(img_path, out_path)
                    except Exception:
                        pass
                stats["error"] += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(all_images)
    print("\n── Results ──────────────────────────────────────────────────")
    print(f"  Total images   : {total}")
    print(f"  Hand cropped   : {stats['cropped']}  ({stats['cropped']/total*100:.1f} %)")
    print(f"  No hand found  : {stats['no_hand']}  ({stats['no_hand']/total*100:.1f} %)")
    print(f"  Errors         : {stats['error']}")
    if copy_on_fail:
        print("  (images with no hand / errors were copied as-is)")
    print(f"\nOutput saved to → {dst_root}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop hands from PSL FingerSpell images using MediaPipe."
    )
    parser.add_argument(
        "--src",
        default="Pakistan_Sign_Language_FingerSpell/dataset",
        help="Path to the source dataset root (default: Pakistan_Sign_Language_FingerSpell/dataset)",
    )
    parser.add_argument(
        "--dst",
        default="PSL_FingerSpell/dataset",
        help="Path to the output dataset root (default: PSL_FingerSpell/dataset)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.15,
        help="Fractional padding around the detected hand bounding box (default: 0.15)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="MediaPipe min_detection_confidence (default: 0.5)",
    )
    parser.add_argument(
        "--no-copy-on-fail",
        action="store_true",
        help="Skip (don't copy) images where no hand is detected instead of copying them as-is",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    if not src.exists():
        print(f"[ERROR] Source path does not exist: {src}")
        sys.exit(1)

    process_dataset(
        src_root=src,
        dst_root=dst,
        padding=args.padding,
        copy_on_fail=not args.no_copy_on_fail,
        min_detection_confidence=args.confidence,
    )