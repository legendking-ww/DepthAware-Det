"""Visualization: HUD, picture-in-picture overlays, layouts."""
from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import matplotlib
import numpy as np

Corner = str  # 'br' | 'bl' | 'tr' | 'tl'


def _resolution_scale(h: int, w: int, ui_scale: float = 1.0) -> float:
    """Base scale from 1080p reference, clamped for readability."""
    base = min(h, w) / 1080.0
    return max(0.85, min(2.2, base * ui_scale))


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    d_min, d_max = d.min(), d.max()
    if d_max - d_min < 1e-6:
        norm = np.zeros_like(d)
    else:
        norm = (d - d_min) / (d_max - d_min)
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_detections(
    image: np.ndarray,
    detections: Sequence[Sequence[float]],
    depths: Sequence[Optional[float]],
    class_names: Union[Mapping[int, str], List[str]],
    ui_scale: float = 1.0,
) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    rs = _resolution_scale(h, w, ui_scale)
    font = 0.55 * rs
    thick = max(2, int(2 * rs))
    lthick = max(1, int(2 * rs))

    for det, depth in zip(detections, depths):
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        name = class_names[cls] if isinstance(class_names, dict) else class_names[cls]
        if depth is not None:
            label = f"{name} {depth:.1f}m"
        else:
            label = f"{name} N/A"
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thick)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, lthick)
        ty = max(int(y1) - 8, th + 8)
        cv2.rectangle(out, (int(x1), ty - th - 6), (int(x1) + tw + 8, ty + 4), (0, 0, 0), -1)
        cv2.putText(
            out, label, (int(x1) + 4, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font, (0, 255, 0), lthick, cv2.LINE_AA,
        )
    return out


def draw_hud(
    frame: np.ndarray,
    scale: float = 1.0,
    fps: Optional[float] = None,
    hints: Optional[Sequence[str]] = None,
    ui_scale: float = 1.0,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    rs = _resolution_scale(h, w, ui_scale)
    bar_h = int((100 if hints else 72) * rs)
    bar_w = min(int(420 * rs), w - 1)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (bar_w, bar_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)

    font_lg = 0.85 * rs
    font_sm = 0.55 * rs
    thick = max(2, int(2 * rs))
    y = int(30 * rs)
    cv2.putText(
        out, f"Scale: {scale:.3f}", (12, y),
        cv2.FONT_HERSHEY_SIMPLEX, font_lg, (0, 255, 255), thick, cv2.LINE_AA,
    )
    y += int(34 * rs)
    if fps is not None:
        cv2.putText(
            out, f"FPS: {fps:.1f}", (12, y),
            cv2.FONT_HERSHEY_SIMPLEX, font_lg, (0, 200, 255), thick, cv2.LINE_AA,
        )
        y += int(32 * rs)
    if hints:
        for line in hints:
            cv2.putText(
                out, line, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_sm, (220, 220, 220), max(1, thick - 1), cv2.LINE_AA,
            )
            y += int(24 * rs)
    return out


def overlay_pip(
    base: np.ndarray,
    panel: np.ndarray,
    corner: Corner = "br",
    margin: int = 14,
    scale_frac: float = 0.34,
    alpha: float = 0.82,
    title: str = "",
    border_color: Tuple[int, int, int] = (255, 255, 255),
    ui_scale: float = 1.0,
) -> np.ndarray:
    out = base.copy()
    h, w = out.shape[:2]
    rs = _resolution_scale(h, w, ui_scale)
    margin = int(margin * rs)
    max_side = int(min(h, w) * scale_frac * rs)
    max_side = max(max_side, int(160 * rs))
    ph, pw = panel.shape[:2]
    s = max_side / max(ph, pw)
    pip_w = max(int(pw * s), int(100 * rs))
    pip_h = max(int(ph * s), int(100 * rs))
    pip = cv2.resize(panel, (pip_w, pip_h), interpolation=cv2.INTER_LINEAR)

    positions = {
        "br": (w - pip_w - margin, h - pip_h - margin),
        "bl": (margin, h - pip_h - margin),
        "tr": (w - pip_w - margin, margin),
        "tl": (margin, margin),
    }
    x0, y0 = positions.get(corner, positions["br"])
    x1, y1 = x0 + pip_w, y0 + pip_h
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return out

    roi = out[y0:y1, x0:x1].astype(np.float32)
    blended = (alpha * pip.astype(np.float32) + (1.0 - alpha) * roi).astype(np.uint8)
    out[y0:y1, x0:x1] = blended

    border = max(2, int(3 * rs))
    cv2.rectangle(out, (x0 - 1, y0 - 1), (x1, y1), border_color, border)
    if title:
        tfont = 0.75 * rs
        tthick = max(2, int(2 * rs))
        ty = y0 + int(28 * rs)
        cv2.putText(
            out, title, (x0 + int(10 * rs), ty),
            cv2.FONT_HERSHEY_SIMPLEX, tfont, (255, 255, 255), tthick + 1, cv2.LINE_AA,
        )
        cv2.putText(
            out, title, (x0 + int(10 * rs), ty),
            cv2.FONT_HERSHEY_SIMPLEX, tfont, (20, 20, 20), tthick, cv2.LINE_AA,
        )
    return out


def compose_realtime_frame(
    vis: np.ndarray,
    depth_map: np.ndarray,
    bev: Optional[np.ndarray] = None,
    show_depth: bool = False,
    scale: float = 1.0,
    fps: Optional[float] = None,
    ui_scale: float = 1.35,
    pip_bev: float = 0.46,
    pip_depth: float = 0.40,
) -> np.ndarray:
    hints = ["[d] depth  [q] quit"]
    if show_depth:
        hints = ["[d] depth ON", "[q] quit"]
    frame = draw_hud(vis, scale=scale, fps=fps, hints=hints, ui_scale=ui_scale)

    if bev is not None:
        frame = overlay_pip(
            frame, bev, corner="br", scale_frac=pip_bev, alpha=0.88,
            title="BEV", ui_scale=ui_scale,
        )
    if show_depth:
        depth_vis = depth_to_colormap(depth_map)
        frame = overlay_pip(
            frame, depth_vis, corner="bl", scale_frac=pip_depth, alpha=0.85,
            title="Depth", ui_scale=ui_scale,
        )
    return frame


def upscale_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    """Upscale output for clearer demo video (does not affect inference)."""
    if target_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w >= target_width:
        return frame
    nh = int(h * target_width / w)
    return cv2.resize(frame, (target_width, nh), interpolation=cv2.INTER_LINEAR)


def compose_static_export(
    vis: np.ndarray,
    depth_map: np.ndarray,
    bev: Optional[np.ndarray] = None,
    scale: float = 1.0,
    fps: Optional[float] = None,
    depth_panel_w: int = 360,
    ui_scale: float = 1.2,
) -> np.ndarray:
    frame = draw_hud(vis, scale=scale, fps=fps, hints=["DepthAware-Det"], ui_scale=ui_scale)
    if bev is not None:
        frame = overlay_pip(
            frame, bev, corner="br", scale_frac=0.42, alpha=0.88, title="BEV", ui_scale=ui_scale,
        )
    h = frame.shape[0]
    depth_vis = depth_to_colormap(depth_map)
    depth_strip = cv2.resize(depth_vis, (depth_panel_w, h))
    cv2.putText(
        depth_strip, "Depth", (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return cv2.hconcat([frame, depth_strip])


def make_dashboard(frame, depth_map, bev=None, **kwargs):
    return compose_static_export(frame, depth_map, bev, **kwargs)
