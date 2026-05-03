"""
Shared utilities for NanoSAM2 live demos.

Imported by torch_camera_predictor.py and onnx_camera_predictor.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------

PALETTE: List[Tuple[int, int, int]] = [
    (0,   200, 255),   # amber-yellow
    (0,   100, 255),   # orange
    (255, 100,   0),   # sky-blue
    (100, 255,   0),   # lime-green
    (255,   0, 150),   # purple
    (0,   255, 200),   # mint-green
    (200,   0, 255),   # magenta
]


# ---------------------------------------------------------------------------
# Mask rendering
# ---------------------------------------------------------------------------

def draw_masks(frame_bgr: np.ndarray, obj_ids: list,
               mask_logits: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay per-object masks on a BGR frame.

    Parameters
    ----------
    frame_bgr   : (H, W, 3) uint8 BGR image.
    obj_ids     : list of integer object ids (1-based, used for colour).
    mask_logits : (N_objs, 1, H, W) float32 numpy array of raw logits.
                  Convert torch tensors with ``torch_masks_to_numpy`` first.
    alpha       : opacity of the coloured overlay (0–1).
    """
    overlay = frame_bgr.copy()
    for i, obj_id in enumerate(obj_ids):
        mask = 1.0 / (1.0 + np.exp(-mask_logits[i, 0]))  # sigmoid → (H, W)
        binary = mask > 0.5

        colour = PALETTE[(obj_id - 1) % len(PALETTE)]
        coloured = np.zeros_like(frame_bgr)
        coloured[binary] = colour
        overlay = cv2.addWeighted(overlay, 1.0, coloured, alpha, 0)

        mask_u8 = binary.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, colour, 1)

    return overlay


def torch_masks_to_numpy(mask_logits) -> np.ndarray:
    """
    Convert PyTorch mask logits to a float32 numpy array of shape (N, 1, H, W).

    Accepts either a (N, 1, H, W) tensor or a list / tuple of (1, H, W) tensors.
    """
    import torch
    if isinstance(mask_logits, (list, tuple)):
        mask_logits = torch.stack(list(mask_logits), dim=0)
    return mask_logits.cpu().float().numpy()


# ---------------------------------------------------------------------------
# Box prompt parsing
# ---------------------------------------------------------------------------

def parse_boxes(box_strings: List[str]) -> List[Tuple[int, int, int, int]]:
    """
    Parse a list of ``'x1,y1,x2,y2'`` strings into (x1, y1, x2, y2) int tuples.

    Raises ``ValueError`` with a descriptive message on bad input.
    """
    result = []
    for s in box_strings:
        parts = s.split(",")
        if len(parts) != 4:
            raise ValueError(
                f"--box expects 'x1,y1,x2,y2' (4 comma-separated integers), got: {s!r}"
            )
        try:
            x1, y1, x2, y2 = [int(p.strip()) for p in parts]
        except ValueError:
            raise ValueError(f"--box values must be integers, got: {s!r}")
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"--box requires x2>x1 and y2>y1, got: {s!r}"
            )
        result.append((x1, y1, x2, y2))
    return result


# ---------------------------------------------------------------------------
# Image-directory frame source
# ---------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def iter_image_dir(directory: Path, loop: bool = True) -> Iterator[np.ndarray]:
    """
    Yield BGR frames from all images in *directory* (sorted by filename).
    If *loop* is True the sequence restarts indefinitely.
    """
    paths = sorted(p for p in directory.iterdir()
                   if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images found in directory: {directory}")
    while True:
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None:
                yield img
        if not loop:
            break


# ---------------------------------------------------------------------------
# Video source management
# ---------------------------------------------------------------------------

def open_video_source(video_str: str):
    """
    Open a camera id, video file path, or image directory.

    Returns
    -------
    cap            : ``cv2.VideoCapture`` or ``None`` when using an image dir.
    image_dir_iter : frame iterator or ``None`` when using camera/video file.
    src            : the original int/str source value (used to detect looping).
    """
    src_path = Path(video_str)
    if src_path.is_dir():
        return None, iter_image_dir(src_path, loop=True), None

    try:
        src = int(video_str)
    except ValueError:
        src = video_str

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {video_str}")
        sys.exit(1)
    return cap, None, src


def read_next_frame(cap, image_dir_iter, src) -> Optional[np.ndarray]:
    """
    Read the next BGR frame from whichever source is active.

    Returns the frame or ``None`` when the source is exhausted.
    Video files are looped automatically; cameras return ``None`` on disconnect.
    """
    if image_dir_iter is not None:
        try:
            return next(image_dir_iter)
        except StopIteration:
            return None

    ret, bgr = cap.read()
    if not ret:
        if isinstance(src, str):          # loop video file
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, bgr = cap.read()
            if ret:
                return bgr
        return None
    return bgr


# ---------------------------------------------------------------------------
# Interactive bounding-box drawing (mouse)
# ---------------------------------------------------------------------------

class BoxDrawer:
    """Draw a bounding box on a CV2 window by clicking and dragging."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start:   Optional[Tuple[int, int]] = None
        self.end:     Optional[Tuple[int, int]] = None
        self.done:    bool = False
        self.drawing: bool = False

    def mouse_cb(self, event, x: int, y: int, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start   = (x, y)
            self.drawing = True
            self.done    = False
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.end     = (x, y)
            self.drawing = False
            self.done    = True

    def draw_on(self, frame: np.ndarray) -> np.ndarray:
        """Draw the current (in-progress or completed) box onto *frame* in-place."""
        if self.start and self.end:
            cv2.rectangle(frame, self.start, self.end, (0, 255, 0), 2)
        return frame

    def get_box(self) -> Optional[Tuple[int, int, int, int]]:
        """Return ``(x1, y1, x2, y2)`` or ``None`` if the box is too small."""
        if self.start is None or self.end is None:
            return None
        x1 = min(self.start[0], self.end[0])
        y1 = min(self.start[1], self.end[1])
        x2 = max(self.start[0], self.end[0])
        y2 = max(self.start[1], self.end[1])
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            return None
        return (x1, y1, x2, y2)
