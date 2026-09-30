"""
📟 Makers - YOLO Object Detection  (wxPython GUI, cross-platform, ONNX Runtime)

Same detection logic/config as the CustomTkinter version. UI notes:

  - Animated splash screen while the default model loads (in a thread).
  - Header: brand + model info/buttons, and a mode switch (Image / Video /
    Webcam / Folder) built from custom-drawn buttons so size and colour are
    identical on Windows, macOS and Linux.
  - Image / Folder pages: display on the left, detection settings +
    analysis on the right in a draggable splitter (2 : 1 by default).
  - Video / Webcam pages: no analysis panel; the view is centred and the
    two detection sliders sit side by side underneath it.

Install:
    pip install wxPython opencv-python numpy onnxruntime
    (for GPU: pip install onnxruntime-gpu)

Packaging with PyInstaller (--onefile):
    The model is looked up via resource_base_dir() (sys._MEIPASS when frozen)
    and runtime output (captures/) goes next to the executable via
    writable_base_dir(). Bundle the model with e.g.:
        pyinstaller --onefile --windowed --add-data "models;models" yolo_wx_app.py   (Windows)
        pyinstaller --onefile --windowed --add-data "models:models" yolo_wx_app.py   (macOS/Linux)
"""

import ast
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import cv2

# On Windows, an unmanifested python.exe is treated as DPI-unaware by
# default: the OS then upscales the whole window with simple bitmap
# stretching to match the display's scale factor, which is what makes
# images (the folder grid especially, being the highest-resolution bitmap
# on screen) look soft/blurry even though nothing in the rendering code is
# actually losing detail. Declaring per-monitor DPI awareness before any
# window is created lets wx hand real physical pixels to the video card
# instead. Must run before wx creates its first window (below).
if sys.platform == 'win32':
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()       # fallback: system DPI aware
    except Exception:
        pass

try:
    import wx
except ImportError:
    print("ERROR: wxPython not installed!")
    print("Install with: pip install wxPython")
    raise SystemExit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed!")
    print("Install with: pip install onnxruntime")
    print("(For GPU acceleration: pip install onnxruntime-gpu)")
    raise SystemExit(1)

# Colorful (light) theme. Key names are kept from the original scheme; the
# extra keys (header / nav / mode / splash) are used by the new layout.
COLORS = {
    # surfaces
    'bg': '#eef2f9',
    'bg_light': '#ffffff',
    'bg_lighter': '#f4f7fd',
    'surface': '#ffffff',
    'canvas': '#dde6f4',
    # text
    'fg': '#1e293b',
    'fg_dim': '#64748b',
    # brand / status
    'accent': '#2563eb',
    'accent_hover': '#1d4ed8',
    'success': '#15803d',
    'warning': '#c2410c',
    'error': '#dc2626',
    'border': '#c3d0e6',
    'danger': '#dc2626',
    'danger_hover': '#b91c1c',
    'green': '#16a34a',
    'green_hover': '#15803d',
    # header (deep indigo bar with light text)
    'header': '#16205a',
    'h_fg': '#ffffff',
    'h_dim': '#b6c2f2',
    'h_accent': '#7dd3fc',
    'h_ok': '#86efac',
    'h_err': '#fca5a5',
    'h_warn': '#fde68a',
    'h_btn': '#3f5bd8',
    'nav_idle_bg': '#2b3d94',
    'nav_idle_fg': '#dbe4ff',
    # one colour per mode
    'mode_image': '#3b82f6',
    'mode_video': '#7c3aed',
    'mode_webcam': '#0d9488',
    'mode_folder': '#ea580c',
    # disabled buttons
    'dis_bg': '#d5dcea',
    'dis_fg': '#8b97ad',
    # splash gradient
    'splash_a': '#1e3a8a',
    'splash_b': '#6d28d9',
}

# A fixed color palette (BGR) used to draw boxes per class id
BOX_PALETTE = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (0, 128, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128),
    (128, 255, 0), (255, 0, 128), (0, 128, 128), (128, 128, 0), (128, 0, 128),
    (192, 192, 192), (64, 64, 255), (64, 255, 64), (255, 64, 64), (200, 200, 0),
]

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
PLACEHOLDER = "📟 Makers - YOLO Object Detection\n\nClick 'Image', 'Video', or 'Webcam' to start"

# Grid settings for folder inference (each cell is rendered at this size)
GRID_COLS, GRID_ROWS = 2, 2
GRID_COUNT = GRID_COLS * GRID_ROWS
CELL_W, CELL_H = 640, 480


# ============================================================
# PyInstaller-safe resource paths
# ============================================================
def resource_base_dir():
    """Base directory for bundled *read-only* resources (models/, etc).

    In a PyInstaller --onefile build, sys._MEIPASS points at the temp folder
    the exe unpacked itself into; use that so a bundled models/ folder is
    found. Running from source, use the folder this script lives in."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def writable_base_dir():
    """Base directory for files this app *creates* at runtime (captures/).

    The --onefile temp extraction folder is wiped after the process exits,
    so runtime output must NOT go there. sys.executable points at the real
    .exe on disk when frozen, so write next to it instead."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


MODELS_DIR = resource_base_dir() / 'models'
CAPTURES_DIR = writable_base_dir() / 'captures'


# ============================================================
# ONNX inference (unchanged logic)
# ============================================================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize + pad image to new_shape while keeping aspect ratio.
    Returns the padded image, the resize ratio, and (dw, dh) half-padding."""
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


class SimpleBox:
    """Mimics the small subset of ultralytics' Boxes API the UI code relies on."""

    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [np.array(xyxy, dtype=float)]


class SimpleResult:
    """Mimics the small subset of ultralytics' Results API the UI code relies on."""

    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class ONNXYOLO:
    """Thin ONNX Runtime wrapper that reproduces the predict() -> result interface
    the rest of the app expects, so the GUI code barely has to change."""

    def __init__(self, model_path):
        providers = []
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        self.session = ort.InferenceSession(model_path, providers=providers)

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.input_h = shape[2] if isinstance(shape[2], int) else 640
        self.input_w = shape[3] if isinstance(shape[3], int) else 640

        self.names = self._load_names()

    def _load_names(self):
        names = {}
        try:
            meta = self.session.get_modelmeta()
            custom = meta.custom_metadata_map
            if custom and 'names' in custom:
                parsed = ast.literal_eval(custom['names'])
                if isinstance(parsed, dict):
                    names = {int(k): v for k, v in parsed.items()}
                elif isinstance(parsed, (list, tuple)):
                    names = {i: v for i, v in enumerate(parsed)}
        except Exception:
            names = {}

        if not names:
            names = {i: f"class{i}" for i in range(1000)}
        return names

    def _postprocess(self, pred, conf_thres, iou_thres, ratio, dw, dh, orig_shape):
        orig_h, orig_w = orig_shape[:2]

        pred = pred[0]
        # Normalize to shape (num_anchors, 4 + num_classes)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes_cxcywh = pred[:, :4]
        class_scores = pred[:, 4:]

        if class_scores.shape[1] == 0:
            return [], [], []

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]

        mask = scores > conf_thres
        boxes_cxcywh = boxes_cxcywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(scores) == 0:
            return [], [], []

        cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Undo letterbox padding/scale to map back to original image coordinates
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        nms_scores = scores.tolist()

        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, conf_thres, iou_thres)
        if indices is None or len(indices) == 0:
            return [], [], []
        indices = np.array(indices).flatten()

        final_boxes = [[float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])] for i in indices]
        final_scores = [float(scores[i]) for i in indices]
        final_class_ids = [int(class_ids[i]) for i in indices]

        return final_boxes, final_scores, final_class_ids

    def _draw(self, img_bgr, boxes):
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = box.cls[0]
            conf = box.conf[0]
            name = self.names.get(cls_id, f"class{cls_id}")
            color = BOX_PALETTE[cls_id % len(BOX_PALETTE)]

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf * 100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty1 = max(0, y1 - th - baseline - 4)
            cv2.rectangle(img_bgr, (x1, ty1), (x1 + tw + 4, ty1 + th + baseline + 4), color, -1)
            cv2.putText(img_bgr, label, (x1 + 2, ty1 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img_bgr

    def infer(self, image_bgr, conf=0.25, iou=0.45):
        """Run detection on a BGR numpy image. Returns (SimpleResult, annotated_bgr)."""
        letter_img, ratio, (dw, dh) = letterbox(image_bgr, (self.input_h, self.input_w))
        img_rgb = cv2.cvtColor(letter_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_chw = np.transpose(img_rgb, (2, 0, 1))
        input_tensor = np.expand_dims(img_chw, 0).astype(np.float32)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        pred = outputs[0]

        boxes, scores, class_ids = self._postprocess(pred, conf, iou, ratio, dw, dh, image_bgr.shape)
        simple_boxes = [SimpleBox(class_ids[i], scores[i], boxes[i]) for i in range(len(boxes))]
        result = SimpleResult(simple_boxes, self.names)

        annotated = self._draw(image_bgr.copy(), simple_boxes)
        return result, annotated


def fit_into_cell(img_bgr, w, h, bg=(45, 45, 45)):
    """Fit an image into a w x h cell (keep aspect ratio, centered)."""
    ih, iw = img_bgr.shape[:2]
    r = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * r)), max(1, int(ih * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    cell = np.full((h, w, 3), bg, dtype=np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    cell[y0:y0 + nh, x0:x0 + nw] = resized
    return cell


def build_grid(annotated_list, names_list):
    """Compose annotated BGR images into a GRID_COLS x GRID_ROWS grid (BGR)."""
    cells = []
    for img, name in zip(annotated_list, names_list):
        cell = fit_into_cell(img, CELL_W, CELL_H)
        label = name if len(name) <= 40 else name[:37] + "..."
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(cell, (0, CELL_H - th - bl - 10), (tw + 12, CELL_H), (30, 30, 30), -1)
        cv2.putText(cell, label, (6, CELL_H - bl - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(cell, (0, 0), (CELL_W - 1, CELL_H - 1), (64, 64, 64), 2)
        cells.append(cell)
    while len(cells) < GRID_COUNT:
        cells.append(np.full((CELL_H, CELL_W, 3), (45, 45, 45), dtype=np.uint8))
    rows = [np.hstack(cells[r * GRID_COLS:(r + 1) * GRID_COLS]) for r in range(GRID_ROWS)]
    return np.vstack(rows)


# ============================================================
# wx helpers
# ============================================================
def C(name):
    """Colour by COLORS key, or by raw '#rrggbb' string."""
    return wx.Colour(COLORS.get(name, name))


def F(size=12, bold=False, family=wx.FONTFAMILY_DEFAULT):
    weight = wx.FONTWEIGHT_BOLD if bold else wx.FONTWEIGHT_NORMAL
    return wx.Font(size, family, wx.FONTSTYLE_NORMAL, weight)


def static_text(parent, text, size=12, bold=False, color='fg', style=0):
    st = wx.StaticText(parent, label=text, style=style)
    st.SetFont(F(size, bold))
    st.SetForegroundColour(C(color))
    return st


class ColorButton(wx.Control):
    """Flat, rounded, custom-drawn button.

    Size, font and colours are fully under our control, so it looks the same
    on Windows / macOS / Linux (native wx.Button ignores colours on some
    platforms and is hard to enlarge). It emits a normal wx.EVT_BUTTON, and
    supports Enable()/SetLabel(), plus a 'toggle' look used by the mode
    switch in the header."""

    def __init__(self, parent, label, handler=None, bg='accent', fg='#ffffff',
                 size=13, height=48, pad_x=20, radius=8, toggle=False,
                 sel_bg=None, sel_fg='#ffffff', idle_bg=None, idle_fg=None):
        super().__init__(parent, style=wx.BORDER_NONE)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)  # required by AutoBufferedPaintDC
        self._label = label
        self._bg = COLORS.get(bg, bg)
        self._fg = COLORS.get(fg, fg)
        self._toggle = toggle
        self._sel_bg = COLORS.get(sel_bg, sel_bg) if sel_bg else self._bg
        self._sel_fg = COLORS.get(sel_fg, sel_fg)
        self._idle_bg = COLORS.get(idle_bg, idle_bg) if idle_bg else self._bg
        self._idle_fg = COLORS.get(idle_fg, idle_fg) if idle_fg else self._fg
        self._selected = False
        self._hover = False
        self._down = False
        self._height = height
        self._pad_x = pad_x
        self._radius = radius
        self._min_w = 0

        self.SetFont(F(size, True))
        self.SetCursor(wx.Cursor(wx.CURSOR_HAND))
        self._update_size()

        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_ENTER_WINDOW, self._on_enter)
        self.Bind(wx.EVT_LEAVE_WINDOW, self._on_leave)
        self.Bind(wx.EVT_LEFT_DOWN, self._on_down)
        self.Bind(wx.EVT_LEFT_DCLICK, self._on_down)
        self.Bind(wx.EVT_LEFT_UP, self._on_up)
        self.Bind(wx.EVT_MOUSE_CAPTURE_LOST, self._on_capture_lost)
        if handler:
            self.Bind(wx.EVT_BUTTON, handler)

    # -- sizing ---------------------------------------------------------
    def _update_size(self, grow_only=False):
        dc = wx.ClientDC(self)
        dc.SetFont(self.GetFont())
        tw, _ = dc.GetTextExtent(self._label)
        w = tw + 2 * self._pad_x
        if grow_only:
            w = max(w, self._min_w)
        changed = (w != self._min_w)
        self._min_w = w
        self.SetMinSize(wx.Size(w, self._height))
        self.InvalidateBestSize()
        return changed

    # -- public API -----------------------------------------------------
    def SetLabel(self, label):
        self._label = label
        if self._update_size(grow_only=True):
            parent = self.GetParent()
            if parent:
                parent.Layout()
        self.Refresh()

    def GetLabel(self):
        return self._label

    def SetSelected(self, selected):
        self._selected = bool(selected)
        self.Refresh()

    def Enable(self, enable=True):
        result = super().Enable(enable)
        self._hover = False
        self._down = False
        self.Refresh()
        return result

    # -- painting -------------------------------------------------------
    def _colors(self):
        if not self.IsEnabled():
            return wx.Colour(COLORS['dis_bg']), wx.Colour(COLORS['dis_fg'])
        if self._toggle:
            base, fg = ((self._sel_bg, self._sel_fg) if self._selected
                        else (self._idle_bg, self._idle_fg))
        else:
            base, fg = self._bg, self._fg
        fill = wx.Colour(base)
        if self._down:
            fill = fill.ChangeLightness(85)
        elif self._hover:
            fill = fill.ChangeLightness(112)
        return fill, wx.Colour(fg)

    def _on_paint(self, evt):
        dc = wx.AutoBufferedPaintDC(self)
        w, h = self.GetClientSize()
        parent = self.GetParent()
        dc.SetBackground(wx.Brush(parent.GetBackgroundColour() if parent else C('bg')))
        dc.Clear()
        fill, fg = self._colors()
        dc.SetBrush(wx.Brush(fill))
        dc.SetPen(wx.Pen(fill))
        dc.DrawRoundedRectangle(0, 0, w, h, self._radius)
        dc.SetFont(self.GetFont())
        dc.SetTextForeground(fg)
        tw, th = dc.GetTextExtent(self._label)
        dc.DrawText(self._label, (w - tw) // 2, (h - th) // 2)

    # -- mouse ----------------------------------------------------------
    def _on_enter(self, evt):
        self._hover = True
        self.Refresh()

    def _on_leave(self, evt):
        self._hover = False
        self.Refresh()

    def _on_down(self, evt):
        if not self.IsEnabled():
            return
        self._down = True
        if not self.HasCapture():
            self.CaptureMouse()
        self.Refresh()

    def _on_up(self, evt):
        was_down = self._down
        self._down = False
        if self.HasCapture():
            self.ReleaseMouse()
        self.Refresh()
        if was_down and self.IsEnabled() and self.GetClientRect().Contains(evt.GetPosition()):
            self._fire()

    def _on_capture_lost(self, evt):
        self._down = False
        self.Refresh()

    def _fire(self):
        evt = wx.CommandEvent(wx.wxEVT_BUTTON, self.GetId())
        evt.SetEventObject(self)
        evt.SetString(self._label)
        self.GetEventHandler().ProcessEvent(evt)


def make_button(parent, label, handler, bg='accent', fg='#ffffff', enabled=True,
                size=13, height=48, pad_x=20):
    btn = ColorButton(parent, label, handler, bg=bg, fg=fg, size=size,
                      height=height, pad_x=pad_x)
    btn.Enable(enabled)
    return btn


# ============================================================
# Splitter that keeps a fixed left:right proportion (default 2:1)
# ============================================================
class RatioSplitter(wx.SplitterWindow):
    """Vertical splitter whose left pane takes `left_ratio` of the width, both
    initially and while the window is resized. The sash stays draggable."""

    def __init__(self, parent, left_ratio=0.67, min_pane=320):
        super().__init__(parent, style=wx.SP_LIVE_UPDATE | wx.SP_3DSASH)
        self._ratio = left_ratio
        self._placed = False
        self.SetMinimumPaneSize(min_pane)
        self.SetSashGravity(left_ratio)
        self.Bind(wx.EVT_SIZE, self._on_size)

    def split(self, left, right):
        self.SplitVertically(left, right)
        wx.CallAfter(self._maybe_place)

    def _on_size(self, evt):
        evt.Skip()
        wx.CallAfter(self._maybe_place)

    def _maybe_place(self):
        try:
            if self._placed or not self.IsSplit():
                return
            width = self.GetClientSize().width
            if width > 400:
                self._placed = True
                self.SetSashPosition(int(width * self._ratio))
        except Exception:
            pass


# ============================================================
# Display panel - replaces the tk.Canvas image/placeholder area
# ============================================================
class DisplayPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent, style=wx.FULL_REPAINT_ON_RESIZE)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)  # required by AutoBufferedPaintDC
        self.SetBackgroundColour(C('canvas'))
        self.SetMinSize((300, 260))
        self._img = None
        self._fill = False
        self.placeholder_visible = True
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_SIZE, lambda e: (self.Refresh(), e.Skip()))

    def show_placeholder(self):
        self.placeholder_visible = True
        self._img = None
        self.Refresh()

    def set_image(self, rgb_array, fill=False):
        """rgb_array: contiguous HxWx3 uint8 numpy array."""
        self.placeholder_visible = False
        rgb_array = np.ascontiguousarray(rgb_array)
        h, w = rgb_array.shape[:2]
        self._img = wx.Image(w, h, rgb_array.tobytes())
        self._fill = fill
        self.Refresh()

    def _on_paint(self, evt):
        dc = wx.AutoBufferedPaintDC(self)
        dc.SetBackground(wx.Brush(C('canvas')))
        dc.Clear()
        cw, ch = self.GetClientSize()
        if cw <= 1 or ch <= 1:
            return

        if self.placeholder_visible or self._img is None:
            dc.SetTextForeground(C('fg_dim'))
            dc.SetFont(F(14))
            lines = PLACEHOLDER.split('\n')
            heights = [dc.GetTextExtent(l)[1] for l in lines]
            total_h = sum(heights) + 4 * (len(lines) - 1)
            y = (ch - total_h) // 2
            for line, lh in zip(lines, heights):
                tw, _ = dc.GetTextExtent(line)
                dc.DrawText(line, max(0, (cw - tw) // 2), y)
                y += lh + 4
        else:
            iw, ih = self._img.GetWidth(), self._img.GetHeight()
            if iw > 0 and ih > 0:
                if self._fill:
                    r = min((cw - 20) / iw, (ch - 20) / ih)
                else:
                    r = min((cw - 20) / iw, (ch - 20) / ih, 1.0)
                nw, nh = max(1, int(iw * r)), max(1, int(ih * r))
                scaled = self._img.Scale(nw, nh, wx.IMAGE_QUALITY_HIGH)
                bmp = wx.Bitmap(scaled)
                dc.DrawBitmap(bmp, (cw - nw) // 2, (ch - nh) // 2, True)

        dc.SetPen(wx.Pen(C('border')))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.DrawRectangle(0, 0, cw, ch)


# ============================================================
# Splash / loading screen
# ============================================================
class SplashFrame(wx.Frame):
    """Frameless animated loading screen: gradient background, the app title,
    a status line and an indeterminate progress bar."""

    W, H = 780, 420
    TITLE = "📟 Makers - YOLO Object Detection"

    def __init__(self):
        super().__init__(None, style=wx.FRAME_NO_TASKBAR | wx.STAY_ON_TOP | wx.NO_BORDER)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetClientSize((self.W, self.H))
        self.Centre()
        self._status = "Loading..."
        self._phase = 0.0
        self._alpha = 255
        self._closing = False

        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_tick, self.timer)
        self.timer.Start(30)

    def set_status(self, text):
        self._status = text
        self.Refresh(False)

    def close(self):
        """Fade out (where the platform supports it) and destroy."""
        if self._closing:
            return
        self._closing = True
        if not self.CanSetTransparent():
            self.timer.Stop()
            self.Destroy()

    def _on_tick(self, evt):
        if self._closing:
            self._alpha -= 30
            if self._alpha <= 0:
                self.timer.Stop()
                self.Destroy()
                return
            try:
                self.SetTransparent(self._alpha)
            except Exception:
                pass
            return
        self._phase = (self._phase + 0.018) % 1.0
        self.Refresh(False)

    def _draw_decor(self, dc, w, h):
        """Soft translucent circles; silently skipped if the platform's
        graphics context is unavailable."""
        try:
            gc = wx.GraphicsContext.Create(dc)
        except Exception:
            return
        if not gc:
            return
        try:
            gc.SetPen(wx.TRANSPARENT_PEN)
            for cx, cy, r, alpha, col in (
                    (w * 0.10, h * 0.16, 130, 40, '#22d3ee'),
                    (w * 0.94, h * 0.88, 180, 36, '#f472b6'),
                    (w * 0.84, h * 0.08, 70, 46, '#facc15')):
                c = wx.Colour(col)
                gc.SetBrush(wx.Brush(wx.Colour(c.Red(), c.Green(), c.Blue(), alpha)))
                gc.DrawEllipse(cx - r, cy - r, 2 * r, 2 * r)
        finally:
            del gc

    def _on_paint(self, evt):
        dc = wx.AutoBufferedPaintDC(self)
        w, h = self.GetClientSize()

        dc.GradientFillLinear(wx.Rect(0, 0, w, h), wx.Colour(COLORS['splash_a']),
                              wx.Colour(COLORS['splash_b']), wx.SOUTH)
        self._draw_decor(dc, w, h)

        # Title (shrinks to fit the window width)
        size = 27
        while True:
            dc.SetFont(F(size, True))
            tw, th = dc.GetTextExtent(self.TITLE)
            if tw <= w - 80 or size <= 14:
                break
            size -= 1
        dc.SetTextForeground(wx.WHITE)
        title_y = int(h * 0.30)
        dc.DrawText(self.TITLE, (w - tw) // 2, title_y)

        # Accent underline
        bar_w = 96
        dc.SetBrush(wx.Brush(wx.Colour('#22d3ee')))
        dc.SetPen(wx.TRANSPARENT_PEN)
        dc.DrawRoundedRectangle((w - bar_w) // 2, title_y + th + 14, bar_w, 5, 2)

        # Indeterminate progress bar
        track_x, track_w, track_h = 130, w - 260, 8
        track_y = h - 110
        dc.SetBrush(wx.Brush(wx.Colour('#5468d4')))
        dc.DrawRoundedRectangle(track_x, track_y, track_w, track_h, 4)
        seg_w = 150
        seg_x = int(track_x - seg_w + (track_w + seg_w) * self._phase)
        dc.SetClippingRegion(wx.Rect(track_x, track_y, track_w, track_h))
        dc.SetBrush(wx.Brush(wx.Colour('#22d3ee')))
        dc.DrawRoundedRectangle(seg_x, track_y, seg_w, track_h, 4)
        dc.DestroyClippingRegion()

        # Status line
        dc.SetFont(F(12))
        dc.SetTextForeground(wx.Colour('#dbe4ff'))
        sw, _ = dc.GetTextExtent(self._status)
        dc.DrawText(self._status, (w - sw) // 2, track_y + 24)

        # Thin outline
        dc.SetPen(wx.Pen(wx.Colour('#8b9cf5')))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.DrawRectangle(0, 0, w, h)


# ============================================================
# Page - holds the widgets/state that belong to one mode
# ============================================================
class Page:
    def __init__(self, mode):
        self.mode = mode
        self.panel = None
        self.display = None
        self.file_label = None
        self.select_btn = None
        self.stop_btn = None
        self.clear_btn = None
        self.save_btn = None
        self.swap_btn = None
        self.shuffle_btn = None
        self.pause_btn = None
        self.progress = None
        self.results_scroll = None   # only Image / Folder pages have a results panel
        self.results_sizer = None
        self.last_result = None


MODES = [
    ('image', "🖼 Image"),
    ('video', "🎬 Video"),
    ('webcam', "📹 Webcam"),
    ('folder', "📂 Folder"),
]
MODE_COLOR = {
    'image': 'mode_image',
    'video': 'mode_video',
    'webcam': 'mode_webcam',
    'folder': 'mode_folder',
}


# ============================================================
# Main frame
# ============================================================
class YOLODetectorFrame(wx.Frame):

    def __init__(self):
        super().__init__(None, title="YOLO Object Detection (ONNX)", size=(1400, 850))
        self.SetMinSize((1100, 700))
        self.SetBackgroundColour(C('bg'))

        # Variables
        self.model = None
        self.model_path = None
        self.current_file = None
        self.folder_path = None
        self.folder_images = []
        self.video_thread = None
        self.stop_video = False
        self.is_processing = False
        self.active_page = None
        self._ui_busy = False
        self.paused = False
        self.conf_value = 0.30
        self.iou_value = 0.30

        # Camera state
        self.available_cameras = [0]
        self.current_cam_index = 0
        self._swapping = False

        self.pages = {}
        self.nav_buttons = {}
        self._conf_ctrls = []   # every copy of the confidence slider (+ value label)
        self._iou_ctrls = []    # every copy of the IoU slider (+ value label)
        self._splash = None
        self._startup_t0 = time.time()
        self._startup_timer = None
        self._dialog_timer = None

        # Gauge pulse timer (drives the indeterminate progress bar)
        self.pulse_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_pulse_timer, self.pulse_timer)

        self.status_bar = self.CreateStatusBar(1)
        self.status_bar.SetStatusText("Ready")

        self._build_ui()
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Centre()

        # Detect cameras in the background so we don't block startup
        threading.Thread(target=self._detect_cameras_bg, daemon=True).start()

    # ------------------------------------------------------------------
    # Startup (splash screen + default model loaded in a worker thread)
    # ------------------------------------------------------------------
    def begin_startup(self, splash):
        self._splash = splash
        self._startup_t0 = time.time()
        splash.set_status("Loading model...")
        self.update_status("Loading default model from models/yolo11n.onnx...", COLORS['warning'])
        threading.Thread(target=self._startup_worker, daemon=True).start()

    def _startup_worker(self):
        model_path = MODELS_DIR / 'yolo11n.onnx'
        model, err = None, None
        if model_path.exists():
            try:
                model = ONNXYOLO(str(model_path))
            except Exception as e:
                err = str(e)
        self.ui(lambda: self._startup_loaded(model_path, model, err))

    def _startup_loaded(self, model_path, model, err):
        # Keep the splash up long enough to be seen, even if the model loads fast
        remaining_ms = max(0, int((1.8 - (time.time() - self._startup_t0)) * 1000))
        self._startup_timer = wx.CallLater(
            remaining_ms, lambda: self._finish_startup(model_path, model, err))

    def _finish_startup(self, model_path, model, err):
        if model is not None:
            self._apply_model(model, str(model_path))
        elif err is not None:
            self.model_label.SetLabel("❌ Failed to load")
            self.model_label.SetForegroundColour(C('h_err'))
            self.update_status("Error loading model", COLORS['error'])
        else:
            self.update_status("No model found in 'models' folder. Please select a model.", COLORS['error'])
            self.model_label.SetLabel("No model loaded")
            self.model_label.SetForegroundColour(C('h_err'))

        self.Maximize(True)
        self.Show()
        self.Raise()
        if self._splash is not None:
            self._splash.close()
            self._splash = None

        # Small delay so the dialog isn't hidden behind the fading splash
        if err is not None:
            self._dialog_timer = wx.CallLater(400, lambda: wx.MessageBox(
                f"Failed to load model:\n{err}", "Error", wx.OK | wx.ICON_ERROR))
        elif model is None:
            self._dialog_timer = wx.CallLater(400, self._ask_for_model)

    def _ask_for_model(self):
        dlg = wx.MessageDialog(
            self, "Model file 'models/yolo11n.onnx' not found.\n\n"
                  "Would you like to select an ONNX model file now?",
            "Model Not Found", wx.YES_NO | wx.ICON_QUESTION)
        if dlg.ShowModal() == wx.ID_YES:
            self.on_load_model(None)
        dlg.Destroy()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = wx.BoxSizer(wx.VERTICAL)
        root.Add(self._build_header(), 0, wx.EXPAND)

        # Thin multi-colour strip: one segment per mode colour
        strip = wx.BoxSizer(wx.HORIZONTAL)
        for mode, _ in MODES:
            seg = wx.Panel(self)
            seg.SetBackgroundColour(C(MODE_COLOR[mode]))
            seg.SetMinSize((1, 4))
            strip.Add(seg, 1, wx.EXPAND)
        root.Add(strip, 0, wx.EXPAND)

        self.workspace_book = wx.Simplebook(self)
        self.workspace_book.SetBackgroundColour(C('bg'))
        for mode, title in MODES:
            page = Page(mode)
            self.pages[mode] = page
            self.workspace_book.AddPage(self._build_page(self.workspace_book, page, title), title)
        root.Add(self.workspace_book, 1, wx.EXPAND)
        self.SetSizer(root)

        self.active_page = self.pages['image']
        self._switch_to_mode('image')

    def _build_header(self):
        hdr = wx.Panel(self)
        hdr.SetBackgroundColour(C('header'))
        outer = wx.BoxSizer(wx.VERTICAL)

        # Row 1: brand (left) + model info and model buttons (right)
        top = wx.BoxSizer(wx.HORIZONTAL)
        brand = wx.BoxSizer(wx.VERTICAL)
        brand.Add(static_text(hdr, "📟 Makers", 22, True, 'h_fg'))
        brand.Add(static_text(hdr, "YOLO Object Detection", 11, False, 'h_dim'))
        top.Add(brand, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.TOP, 18)
        top.AddStretchSpacer(1)

        info = wx.BoxSizer(wx.VERTICAL)
        info.Add(static_text(hdr, "🤖 ONNX Model", 12, True, 'h_accent'))
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(static_text(hdr, "Current Model:", 11, False, 'h_dim'), 0,
                wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 12)
        self.model_label = static_text(hdr, "Loading...", 13, True, 'h_warn',
                                       style=wx.ST_ELLIPSIZE_MIDDLE)
        self.model_label.SetMinSize((320, -1))
        row.Add(self.model_label, 0, wx.ALIGN_CENTER_VERTICAL)
        info.Add(row, 0, wx.TOP, 4)
        top.Add(info, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 36)

        top.Add(make_button(hdr, "📂 Change Model", self.on_load_model, bg='h_btn',
                            size=12, height=42, pad_x=18),
                0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        top.Add(make_button(hdr, "🔄 Reload", self.on_reload_model, bg='h_btn',
                            size=12, height=42, pad_x=18),
                0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 18)
        outer.Add(top, 0, wx.EXPAND | wx.TOP, 10)

        # Row 2: mode switch (radio group)
        nav = wx.BoxSizer(wx.HORIZONTAL)
        nav.Add(static_text(hdr, "📁 Select Option", 13, True, 'h_accent'), 0,
                wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 18)
        for mode, title in MODES:
            btn = ColorButton(hdr, title, bg='nav_idle_bg', fg='nav_idle_fg', size=14,
                              height=46, pad_x=26, toggle=True,
                              sel_bg=MODE_COLOR[mode], sel_fg='#ffffff',
                              idle_bg='nav_idle_bg', idle_fg='nav_idle_fg')
            btn.Bind(wx.EVT_BUTTON, lambda e, m=mode: self._switch_to_mode(m))
            self.nav_buttons[mode] = btn
            nav.Add(btn, 0, wx.RIGHT, 8)
        outer.Add(nav, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 12)

        hdr.SetSizer(outer)
        return hdr

    def _switch_to_mode(self, mode):
        idx = [m for m, _ in MODES].index(mode)
        # Switching pages while a stream is running: stop it first
        if self.is_processing and self.active_page and self.active_page.mode != mode:
            self.stop_video = True
        self.workspace_book.SetSelection(idx)
        self.active_page = self.pages[mode]
        for m, btn in self.nav_buttons.items():
            btn.SetSelected(m == mode)

    # -- pages ---------------------------------------------------------
    def _build_page(self, book, page, title):
        if page.mode in ('image', 'folder'):
            # Display on the left, settings + analysis on the right (2 : 1)
            splitter = RatioSplitter(book, left_ratio=0.67, min_pane=320)
            splitter.SetBackgroundColour(C('bg'))
            left = self._build_workspace(splitter, page, title, centered=False)
            right = self._build_side_panel(splitter, page)
            splitter.split(left, right)
            return splitter
        # Video / Webcam: centred view with settings underneath, no analysis panel
        return self._build_workspace(book, page, title, centered=True)

    def _make_threshold(self, parent, caption, kind, width=None):
        """One threshold control (caption + value + slider). Every copy is
        registered so all sliders in the app stay in sync."""
        box = wx.BoxSizer(wx.VERTICAL)
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(static_text(parent, caption, 12), 1, wx.ALIGN_CENTER_VERTICAL)
        value = static_text(parent, "0.30", 13, True, 'accent')
        row.Add(value, 0, wx.ALIGN_CENTER_VERTICAL)
        box.Add(row, 0, wx.EXPAND)

        slider = wx.Slider(parent, minValue=10, maxValue=90, value=30, style=wx.SL_HORIZONTAL)
        if width:
            slider.SetMinSize((width, -1))
        if kind == 'conf':
            slider.Bind(wx.EVT_SLIDER, self.on_conf_slider)
            self._conf_ctrls.append((slider, value))
        else:
            slider.Bind(wx.EVT_SLIDER, self.on_iou_slider)
            self._iou_ctrls.append((slider, value))
        box.Add(slider, 0, wx.EXPAND | wx.TOP, 6)
        return box

    def _build_workspace(self, parent, page, title, centered):
        mode = page.mode
        panel = wx.Panel(parent)
        panel.SetBackgroundColour(C('bg'))
        page.panel = panel
        outer = wx.BoxSizer(wx.VERTICAL)

        # Selected-file bar
        info = wx.Panel(panel)
        info.SetBackgroundColour(C('surface'))
        isz = wx.BoxSizer(wx.HORIZONTAL)
        if centered:
            isz.AddStretchSpacer(1)
        isz.Add(static_text(info, "📄 Selected File:", 12, False, 'fg_dim'), 0,
                wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.TOP | wx.BOTTOM, 14)
        page.file_label = static_text(info, "No file selected", 13, True, 'error')
        isz.Add(page.file_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 8)
        if centered:
            isz.AddStretchSpacer(1)
        info.SetSizer(isz)
        outer.Add(info, 0, wx.EXPAND | wx.ALL, 10)

        # Action buttons
        actions = wx.BoxSizer(wx.HORIZONTAL)
        if centered:
            actions.AddStretchSpacer(1)

        def add(btn):
            actions.Add(btn, 0, wx.LEFT | wx.RIGHT, 4)

        select_cmd = {
            'image': self.upload_image,
            'video': self.upload_video,
            'webcam': self.use_webcam,
            'folder': self.upload_folder,
        }[mode]
        page.select_btn = make_button(panel, title, select_cmd, bg=MODE_COLOR[mode])
        add(page.select_btn)

        if mode == 'folder':
            page.shuffle_btn = make_button(panel, "🎲 Shuffle 4 Images", self.on_shuffle_folder,
                                           enabled=False)
            add(page.shuffle_btn)

        if mode == 'webcam':
            page.swap_btn = make_button(panel, "🔀 Swap Cam", self.on_swap_camera, enabled=False)
            add(page.swap_btn)

        if mode == 'video':
            page.pause_btn = make_button(panel, "⏸ Pause", self.on_toggle_pause, enabled=False)
            add(page.pause_btn)

        if mode in ('video', 'webcam'):
            page.stop_btn = make_button(panel, "⏹ Stop", self.on_stop_detection,
                                        bg='danger', enabled=False)
            add(page.stop_btn)

        page.clear_btn = make_button(panel, "🗑 Clear", self.on_clear_display, bg='danger')
        add(page.clear_btn)

        if mode == 'webcam':
            page.save_btn = make_button(panel, "📸 Capture", self.on_capture_frame,
                                        bg='green', enabled=False)
            add(page.save_btn)
        elif mode != 'video':
            page.save_btn = make_button(panel, "💾 Save Result", self.on_save_result,
                                        bg='green', enabled=False)
            add(page.save_btn)

        if centered:
            actions.AddStretchSpacer(1)
        outer.Add(actions, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Display
        outer.Add(static_text(panel, "📺 Display", 13, True, 'accent'), 0,
                  wx.ALIGN_CENTER_HORIZONTAL if centered else wx.LEFT, 0 if centered else 12)
        page.display = DisplayPanel(panel)
        outer.Add(page.display, 1, wx.EXPAND | wx.ALL, 10)

        page.progress = wx.Gauge(panel, range=100, style=wx.GA_HORIZONTAL)
        page.progress.Hide()
        outer.Add(page.progress, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Video / Webcam: detection settings under the view, sliders side by side
        if centered:
            card = wx.Panel(panel)
            card.SetBackgroundColour(C('surface'))
            csz = wx.BoxSizer(wx.VERTICAL)
            csz.Add(static_text(card, "⚙️ Detection Settings", 13, True, 'accent'), 0,
                    wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, 12)
            row = wx.BoxSizer(wx.HORIZONTAL)
            row.Add(self._make_threshold(card, "Confidence Threshold:", 'conf', width=380),
                    0, wx.ALL, 14)
            row.Add(self._make_threshold(card, "IoU Threshold:", 'iou', width=380),
                    0, wx.ALL, 14)
            csz.Add(row, 0, wx.ALIGN_CENTER_HORIZONTAL)
            card.SetSizer(csz)
            outer.Add(card, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.BOTTOM, 12)

        panel.SetSizer(outer)
        return panel

    def _build_side_panel(self, parent, page):
        """Right-hand pane of the Image / Folder pages: collapsible detection
        settings on top, detection analysis underneath."""
        panel = wx.Panel(parent)
        panel.SetBackgroundColour(C('bg'))
        sizer = wx.BoxSizer(wx.VERTICAL)

        pane = wx.CollapsiblePane(panel, label="⚙️ Detection Settings",
                                  style=wx.CP_DEFAULT_STYLE | wx.CP_NO_TLW_RESIZE)
        pane.SetBackgroundColour(C('surface'))
        pane.SetFont(F(12, True))
        inner = pane.GetPane()
        inner.SetBackgroundColour(C('surface'))
        isz = wx.BoxSizer(wx.VERTICAL)
        isz.Add(self._make_threshold(inner, "Confidence Threshold:", 'conf'),
                0, wx.EXPAND | wx.ALL, 10)
        isz.Add(self._make_threshold(inner, "IoU Threshold:", 'iou'),
                0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        inner.SetSizer(isz)
        pane.Collapse(False)
        pane.Bind(wx.EVT_COLLAPSIBLEPANE_CHANGED,
                  lambda e: (panel.Layout(), panel.Refresh()))
        sizer.Add(pane, 0, wx.EXPAND | wx.ALL, 8)

        banner = wx.Panel(panel)
        banner.SetBackgroundColour(C('accent'))
        bsz = wx.BoxSizer(wx.HORIZONTAL)
        bsz.Add(static_text(banner, "📊 Detection Results", 13, True, '#ffffff'), 0, wx.ALL, 10)
        banner.SetSizer(bsz)
        sizer.Add(banner, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        page.results_scroll = wx.ScrolledWindow(panel, style=wx.VSCROLL)
        page.results_scroll.SetScrollRate(0, 12)
        page.results_scroll.SetBackgroundColour(C('bg'))
        page.results_sizer = wx.BoxSizer(wx.VERTICAL)
        page.results_scroll.SetSizer(page.results_sizer)
        sizer.Add(page.results_scroll, 1, wx.EXPAND | wx.ALL, 8)

        panel.SetSizer(sizer)
        self.show_no_results(page)
        return panel

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def ui(self, fn):
        """Schedule fn on the UI thread (safe to call from worker threads)."""
        try:
            wx.CallAfter(fn)
        except Exception:
            pass

    def on_conf_slider(self, evt):
        v = evt.GetEventObject().GetValue()
        self.conf_value = v / 100.0
        for slider, label in self._conf_ctrls:
            if slider.GetValue() != v:
                slider.SetValue(v)
            label.SetLabel(f"{self.conf_value:.2f}")

    def on_iou_slider(self, evt):
        v = evt.GetEventObject().GetValue()
        self.iou_value = v / 100.0
        for slider, label in self._iou_ctrls:
            if slider.GetValue() != v:
                slider.SetValue(v)
            label.SetLabel(f"{self.iou_value:.2f}")

    def update_status(self, message, color=None):
        self.status_bar.SetStatusText(message)

    def _set_file_label(self, page, text, color):
        page.file_label.SetLabel(text)
        page.file_label.SetForegroundColour(C(color))
        page.file_label.GetParent().Layout()
        page.file_label.Refresh()

    def _require_model(self):
        if not self.model:
            wx.MessageBox("Please load a model first!", "Warning",
                          wx.OK | wx.ICON_WARNING)
            return False
        return True

    def _on_pulse_timer(self, evt):
        if self.active_page and self.active_page.progress.IsShown():
            self.active_page.progress.Pulse()

    def _begin_job(self, page, show_progress):
        self.is_processing = True
        self.stop_video = False
        page.select_btn.Enable(False)
        if page.shuffle_btn:
            page.shuffle_btn.Enable(False)
        if page.stop_btn:
            page.stop_btn.Enable(True)
        if page.pause_btn:
            self.paused = False
            page.pause_btn.Enable(True)
            page.pause_btn.SetLabel("⏸ Pause")
        if page.mode == 'webcam':
            page.save_btn.Enable(True)
        if show_progress:
            page.progress.Show()
            page.progress.Pulse()
            page.panel.Layout()
            self.pulse_timer.Start(100)

    def _end_job(self, page):
        page.progress.Hide()
        self.pulse_timer.Stop()
        page.panel.Layout()
        page.select_btn.Enable(True)
        if page.shuffle_btn and self.folder_path:
            page.shuffle_btn.Enable(True)
        if page.stop_btn:
            page.stop_btn.Enable(False)
        if page.pause_btn:
            self.paused = False
            page.pause_btn.Enable(False)
            page.pause_btn.SetLabel("⏸ Pause")
        self.is_processing = False

    # ------------------------------------------------------------------
    # Camera detection / swapping
    # ------------------------------------------------------------------
    def _detect_cameras_bg(self):
        """Probe a few camera indices in a background thread."""
        found = []
        for i in range(3):
            try:
                cap = cv2.VideoCapture(i)
                if cap is not None and cap.isOpened():
                    found.append(i)
                cap.release()
            except Exception:
                pass
        if not found:
            found = [0]
        self.ui(lambda: self._on_cameras_detected(found))

    def _on_cameras_detected(self, cams):
        self.available_cameras = cams
        self.current_cam_index = cams[0]
        self.pages['webcam'].swap_btn.Enable(len(cams) > 1)

    def on_swap_camera(self, evt):
        self.swap_camera()

    def swap_camera(self):
        """Cycle to the next detected camera index.

        Clean stop -> wait for the capture to fully release -> restart, so the
        old and new camera handles never overlap."""
        if len(self.available_cameras) < 2:
            return
        if self._swapping:
            return

        page = self.pages['webcam']
        try:
            idx = self.available_cameras.index(self.current_cam_index)
        except ValueError:
            idx = -1
        idx = (idx + 1) % len(self.available_cameras)
        new_index = self.available_cameras[idx]
        self.current_cam_index = new_index

        if not isinstance(self.current_file, int):
            self.update_status(
                f"Camera set to index {new_index} (used next time you select Webcam)",
                COLORS['accent'])
            return

        self._swapping = True
        page.swap_btn.Enable(False)

        if self.is_processing:
            self.update_status(f"Switching to camera {new_index}...", COLORS['warning'])
            self.stop_video = True
            old_thread = self.video_thread
            threading.Thread(target=self._wait_and_restart_camera,
                             args=(old_thread, new_index), daemon=True).start()
        else:
            self.current_file = new_index
            self._set_file_label(page, f"📹 Webcam {new_index}", 'success')
            self.update_status(f"Switched to camera index {new_index}", COLORS['accent'])
            self._swapping = False
            page.swap_btn.Enable(True)

    def _wait_and_restart_camera(self, old_thread, new_index):
        if old_thread is not None:
            old_thread.join(timeout=5)
        self.ui(lambda: self._finish_camera_swap(new_index))

    def _finish_camera_swap(self, new_index):
        page = self.pages['webcam']
        self.current_file = new_index
        self._set_file_label(page, f"📹 Webcam {new_index}", 'success')
        self._swapping = False
        page.swap_btn.Enable(len(self.available_cameras) > 1)
        self.update_status(f"Switched to camera index {new_index}", COLORS['accent'])
        # Seamlessly resume detection on the new camera
        self.run_video(page, self.current_file)

    # ------------------------------------------------------------------
    # Model loading (ONNX)
    # ------------------------------------------------------------------
    def _apply_model(self, model, file_path):
        self.model = model
        self.model_path = file_path
        model_name = Path(file_path).name
        self.model_label.SetLabel(f"✓ {model_name}")
        self.model_label.SetForegroundColour(C('h_ok'))
        self.model_label.SetToolTip(str(file_path))
        self.model_label.Refresh()
        self.update_status(f"Model loaded successfully: {model_name}", COLORS['success'])

    def on_load_model(self, evt):
        """Load YOLO ONNX model from file dialog"""
        initial_dir = str(MODELS_DIR) if MODELS_DIR.exists() else "."
        dlg = wx.FileDialog(self, message="Select YOLO ONNX Model File", defaultDir=initial_dir,
                            wildcard="ONNX Models (*.onnx)|*.onnx|All files (*.*)|*.*",
                            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            self.load_model_file(dlg.GetPath())
        dlg.Destroy()

    def load_model_file(self, file_path):
        """Load ONNX model from given path"""
        try:
            self.update_status("Loading model...", COLORS['warning'])
            wx.Yield()
            model = ONNXYOLO(file_path)
            self._apply_model(model, file_path)

        except Exception as e:
            wx.MessageBox(f"Failed to load model:\n{str(e)}", "Error", wx.OK | wx.ICON_ERROR)
            self.model_label.SetLabel("❌ Failed to load")
            self.model_label.SetForegroundColour(C('h_err'))
            self.update_status("Error loading model", COLORS['error'])

    def on_reload_model(self, evt):
        """Reload the current model"""
        if self.model_path:
            self.load_model_file(self.model_path)
        else:
            wx.MessageBox("No model to reload. Please load a model first.", "Info",
                          wx.OK | wx.ICON_INFORMATION)

    # ------------------------------------------------------------------
    # Selection actions - each one starts detection immediately
    # ------------------------------------------------------------------
    def upload_image(self, evt):
        """Select an image and run detection immediately"""
        if not self._require_model():
            return
        dlg = wx.FileDialog(
            self, message="Select Image",
            wildcard="Image files (*.jpg;*.jpeg;*.png;*.bmp)|*.jpg;*.jpeg;*.png;*.bmp|"
                     "All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dlg.ShowModal() != wx.ID_OK:
            dlg.Destroy()
            return
        file_path = dlg.GetPath()
        dlg.Destroy()

        page = self.pages['image']
        self.current_file = file_path
        self._set_file_label(page, Path(file_path).name, 'success')
        self.update_status(f"Image loaded: {Path(file_path).name}", COLORS['success'])
        self.run_image(page, file_path)

    def upload_video(self, evt):
        """Select a video and run detection immediately"""
        if not self._require_model():
            return
        dlg = wx.FileDialog(
            self, message="Select Video",
            wildcard="Video files (*.mp4;*.avi;*.mov;*.mkv)|*.mp4;*.avi;*.mov;*.mkv|"
                     "All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dlg.ShowModal() != wx.ID_OK:
            dlg.Destroy()
            return
        file_path = dlg.GetPath()
        dlg.Destroy()

        page = self.pages['video']
        self.current_file = file_path
        self._set_file_label(page, Path(file_path).name, 'success')
        self.update_status(f"Video loaded: {Path(file_path).name}", COLORS['success'])
        self.run_video(page, file_path)

    def use_webcam(self, evt):
        """Use webcam for detection - starts immediately"""
        if not self._require_model():
            return
        page = self.pages['webcam']
        self.current_file = self.current_cam_index
        self._set_file_label(page, f"📹 Webcam {self.current_cam_index}", 'success')
        self.update_status(f"Webcam {self.current_cam_index} selected", COLORS['success'])
        self.run_video(page, self.current_file)

    def upload_folder(self, evt):
        """Select a folder, randomly pick 4 images and run batch detection"""
        if not self._require_model():
            return
        dlg = wx.DirDialog(self, message="Select Image Folder")
        if dlg.ShowModal() != wx.ID_OK:
            dlg.Destroy()
            return
        folder = dlg.GetPath()
        dlg.Destroy()

        images = [p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not images:
            wx.MessageBox("No images found in the selected folder!", "Warning",
                          wx.OK | wx.ICON_WARNING)
            return

        page = self.pages['folder']
        self.folder_path = folder
        self.folder_images = images
        self._set_file_label(page, f"{Path(folder).name}  ({len(images)} images)", 'success')
        self.update_status(f"Folder loaded: {Path(folder).name} ({len(images)} images)",
                           COLORS['success'])
        self.shuffle_folder()

    def on_shuffle_folder(self, evt):
        self.shuffle_folder()

    def shuffle_folder(self):
        """Pick a new random set of up to 4 images and run detection"""
        if not self._require_model() or not self.folder_images:
            return
        page = self.pages['folder']
        picks = random.sample(self.folder_images, min(GRID_COUNT, len(self.folder_images)))
        self.run_folder(page, picks)

    # ------------------------------------------------------------------
    # Image inference
    # ------------------------------------------------------------------
    def run_image(self, page, path):
        if self.is_processing:
            return
        self._begin_job(page, show_progress=True)
        self.update_status("Processing image...", COLORS['warning'])
        conf, iou = self.conf_value, self.iou_value
        threading.Thread(target=self._image_worker, args=(page, path, conf, iou),
                         daemon=True).start()

    def _image_worker(self, page, path, conf, iou):
        try:
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                raise ValueError("Could not read the selected image file")
            result, annotated_bgr = self.model.infer(img_bgr, conf=conf, iou=iou)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            self.ui(lambda: self._image_done(page, result, annotated_rgb))
        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = str(e)
            self.ui(lambda: self._job_failed(page, "Detection failed", msg))

    def _image_done(self, page, result, annotated_rgb):
        page.display.set_image(annotated_rgb, fill=False)
        self.display_results(page, result)
        page.last_result = annotated_rgb
        page.save_btn.Enable(True)
        self.update_status(f"✓ Detection complete: {len(result.boxes)} objects found",
                           COLORS['success'])
        self._end_job(page)

    def _job_failed(self, page, status, detail):
        wx.MessageBox(f"{status}:\n{detail}", "Error", wx.OK | wx.ICON_ERROR)
        self.update_status(status, COLORS['error'])
        self._end_job(page)

    # ------------------------------------------------------------------
    # Folder (batch) inference -> 2x2 grid
    # ------------------------------------------------------------------
    def run_folder(self, page, paths):
        if self.is_processing:
            return
        self._begin_job(page, show_progress=True)
        self.update_status(f"Processing {len(paths)} images...", COLORS['warning'])
        conf, iou = self.conf_value, self.iou_value
        threading.Thread(target=self._folder_worker, args=(page, paths, conf, iou),
                         daemon=True).start()

    def _folder_worker(self, page, paths, conf, iou):
        try:
            entries, annotated_list, names_list = [], [], []
            for p in paths:
                img_bgr = cv2.imread(str(p))
                if img_bgr is None:
                    continue
                result, annotated_bgr = self.model.infer(img_bgr, conf=conf, iou=iou)
                entries.append((p.name, result))
                annotated_list.append(annotated_bgr)
                names_list.append(p.name)
            if not entries:
                raise ValueError("None of the selected images could be read")

            grid_bgr = build_grid(annotated_list, names_list)
            grid_rgb = cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB)
            self.ui(lambda: self._folder_done(page, entries, grid_rgb))
        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = str(e)
            self.ui(lambda: self._job_failed(page, "Batch detection failed", msg))

    def _folder_done(self, page, entries, grid_rgb):
        page.display.set_image(grid_rgb, fill=False)
        self.display_folder_results(page, entries)
        page.last_result = grid_rgb
        page.save_btn.Enable(True)
        total = sum(len(r.boxes) for _, r in entries)
        self.update_status(
            f"✓ Detection complete: {total} objects found in {len(entries)} images",
            COLORS['success'])
        self._end_job(page)

    # ------------------------------------------------------------------
    # Video / webcam inference
    # ------------------------------------------------------------------
    def run_video(self, page, source):
        if self.is_processing:
            return
        self._begin_job(page, show_progress=False)
        self.video_thread = threading.Thread(target=self._video_worker,
                                             args=(page, source), daemon=True)
        self.video_thread.start()

    def _push_frame(self, page, rgb, text):
        """Drop frames if the UI hasn't finished drawing the previous one."""
        if self._ui_busy:
            return
        self._ui_busy = True
        self.ui(lambda: self._show_frame(page, rgb, text))

    def _show_frame(self, page, rgb, text):
        try:
            page.display.set_image(rgb, fill=True)
            page.last_result = rgb
            self.update_status(text, COLORS['warning'])
        finally:
            self._ui_busy = False

    def _video_worker(self, page, source):
        frame_count = 0
        total_detections = 0
        error = None
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self.ui(lambda: wx.MessageBox("Failed to open video source!", "Error",
                                              wx.OK | wx.ICON_ERROR))
                cap.release()
                self.ui(lambda: self._video_done(page, 0, 0, True, None))
                return

            while cap.isOpened() and not self.stop_video:
                if self.paused:
                    time.sleep(0.05)
                    continue
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                result, annotated_bgr = self.model.infer(
                    frame, conf=self.conf_value, iou=self.iou_value)
                total_detections += len(result.boxes)

                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                self._push_frame(
                    page, annotated_rgb,
                    f"Frame {frame_count} - {len(result.boxes)} objects detected")

                time.sleep(0.01)

            cap.release()
        except Exception as e:
            error = str(e)

        stopped = self.stop_video
        self.ui(lambda: self._video_done(page, frame_count, total_detections, stopped, error))

    def _video_done(self, page, frame_count, total_detections, stopped, error):
        self._ui_busy = False
        self._end_job(page)
        if page.mode == 'webcam' and page.save_btn:
            page.save_btn.Enable(False)

        if error:
            wx.MessageBox(f"Video processing failed:\n{error}", "Error", wx.OK | wx.ICON_ERROR)
            self.update_status("Video processing failed", COLORS['error'])
        elif self._swapping:
            pass  # camera swap in progress; status handled there
        elif not stopped:
            self.update_status(
                f"✓ Video complete: {frame_count} frames, {total_detections} total detections",
                COLORS['success'])
            wx.MessageBox(f"Video processing complete!\n"
                          f"Frames: {frame_count}\n"
                          f"Total detections: {total_detections}", "Complete",
                          wx.OK | wx.ICON_INFORMATION)
        else:
            self.update_status("Video processing stopped", COLORS['warning'])

    def on_toggle_pause(self, evt):
        """Play/pause toggle for video playback"""
        page = self.pages['video']
        if not self.is_processing:
            return
        self.paused = not self.paused
        if self.paused:
            page.pause_btn.SetLabel("▶ Play")
            self.update_status("Video paused", COLORS['warning'])
        else:
            page.pause_btn.SetLabel("⏸ Pause")
            self.update_status("Video playing", COLORS['accent'])

    def on_capture_frame(self, evt):
        """Save the current annotated webcam frame instantly (no dialog)"""
        page = self.pages['webcam']
        if page.last_result is None:
            wx.MessageBox("No frame to capture yet!", "Warning", wx.OK | wx.ICON_WARNING)
            return
        try:
            CAPTURES_DIR.mkdir(exist_ok=True)
            file_path = CAPTURES_DIR / f"capture_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
            frame = page.last_result.copy()
            cv2.imwrite(str(file_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.update_status(f"📸 Captured: {file_path}", COLORS['success'])
        except Exception as e:
            wx.MessageBox(f"Failed to capture frame:\n{str(e)}", "Error", wx.OK | wx.ICON_ERROR)

    def on_stop_detection(self, evt):
        """Stop video processing"""
        self.paused = False
        self.stop_video = True
        self.update_status("Stopping...", COLORS['warning'])

    # ------------------------------------------------------------------
    # Results panel (Image / Folder pages only)
    # ------------------------------------------------------------------
    def _clear_results(self, page):
        if page.results_sizer is None:
            return
        page.results_sizer.Clear(delete_windows=True)

    def _refresh_results(self, page):
        if page.results_sizer is None:
            return
        page.results_sizer.Layout()
        page.results_scroll.FitInside()
        page.results_scroll.Refresh()

    def show_no_results(self, page):
        if page.results_sizer is None:
            return
        self._clear_results(page)
        hint = {
            'image': "No detections yet\n\nSelect an image\nto run detection",
            'video': "No detections yet\n\nSelect a video\nto run detection",
            'webcam': "No detections yet\n\nStart the webcam\nto run detection",
            'folder': "No detections yet\n\nSelect a folder\nto run batch detection",
        }[page.mode]
        st = static_text(page.results_scroll, hint, 12, False, 'fg_dim')
        page.results_sizer.Add(st, 0, wx.ALL | wx.ALIGN_CENTER, 50)
        self._refresh_results(page)

    def _make_card(self, page, accent='accent', orient=wx.VERTICAL):
        """White card with a coloured stripe on its left edge. Returns
        (card, inner_sizer); add content to inner_sizer, then add the card
        to page.results_sizer."""
        card = wx.Panel(page.results_scroll)
        card.SetBackgroundColour(C('surface'))
        outer = wx.BoxSizer(wx.HORIZONTAL)
        strip = wx.Panel(card)
        strip.SetBackgroundColour(C(accent))
        strip.SetMinSize((5, 1))
        outer.Add(strip, 0, wx.EXPAND)
        inner = wx.BoxSizer(orient)
        outer.Add(inner, 1, wx.EXPAND)
        card.SetSizer(outer)
        return card, inner

    def _total_card(self, page, count, subtitle="Objects Detected"):
        card, sizer = self._make_card(page, 'accent')
        sizer.Add(static_text(card, f"{count}", 36, True, 'accent'), 0,
                  wx.ALIGN_CENTER | wx.TOP, 12)
        sizer.Add(static_text(card, subtitle, 11, False, 'fg_dim'), 0,
                  wx.ALIGN_CENTER | wx.BOTTOM, 12)
        page.results_sizer.Add(card, 0, wx.EXPAND | wx.ALL, 4)

    def _section_header(self, page, text):
        st = static_text(page.results_scroll, text, 13, True, 'accent')
        page.results_sizer.Add(st, 0, wx.EXPAND | wx.LEFT | wx.TOP, 8)

    def _divider(self, page):
        line = wx.StaticLine(page.results_scroll)
        page.results_sizer.Add(line, 0, wx.EXPAND | wx.ALL, 12)

    def display_results(self, page, result, frame_num=None):
        """Display detection results in structured format"""
        self._clear_results(page)
        boxes = result.boxes

        if frame_num:
            card, sizer = self._make_card(page, 'mode_video')
            sizer.Add(static_text(card, f"FRAME {frame_num}", 12, True, 'accent'), 0,
                      wx.ALIGN_CENTER | wx.ALL, 8)
            page.results_sizer.Add(card, 0, wx.EXPAND | wx.ALL, 4)

        self._total_card(page, len(boxes))

        if len(boxes) == 0:
            page.results_sizer.Add(static_text(page.results_scroll, "No objects detected", 11,
                                               False, 'fg_dim'), 0, wx.ALIGN_CENTER | wx.ALL, 20)
            self._refresh_results(page)
            return

        self._section_header(page, "📈 Summary by Class")
        class_counts = {}
        for box in boxes:
            class_name = result.names[int(box.cls[0])]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            self.create_summary_card(page, class_name, count)

        self._divider(page)

        self._section_header(page, "🔍 Detailed Detections")
        for i, box in enumerate(boxes, 1):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0]
            self.create_detection_card(page, i, class_name, conf, x1, y1, x2, y2)

        self._refresh_results(page)

    def display_folder_results(self, page, entries):
        """Analysis for the 2x2 grid: overall totals, class summary across all
        images, then one card per image (count, avg/max confidence, classes)."""
        self._clear_results(page)

        total = sum(len(r.boxes) for _, r in entries)
        all_confs = [float(b.conf[0]) for _, r in entries for b in r.boxes]
        avg_conf = (sum(all_confs) / len(all_confs)) if all_confs else 0.0

        self._total_card(page, total, f"Objects Detected in {len(entries)} Images")

        if total == 0:
            page.results_sizer.Add(static_text(page.results_scroll, "No objects detected", 11,
                                               False, 'fg_dim'), 0, wx.ALIGN_CENTER | wx.ALL, 20)
        else:
            card, sizer = self._make_card(page, 'success', orient=wx.HORIZONTAL)
            for label, value in (("Avg Confidence", f"{avg_conf * 100:.1f}%"),
                                 ("Max Confidence", f"{max(all_confs) * 100:.1f}%"),
                                 ("Avg / Image", f"{total / len(entries):.1f}")):
                col = wx.BoxSizer(wx.VERTICAL)
                col.Add(static_text(card, value, 14, True, 'success'), 0, wx.ALIGN_CENTER)
                col.Add(static_text(card, label, 9, False, 'fg_dim'), 0, wx.ALIGN_CENTER)
                sizer.Add(col, 1, wx.ALIGN_CENTER | wx.ALL, 10)
            page.results_sizer.Add(card, 0, wx.EXPAND | wx.ALL, 4)

            self._section_header(page, "📈 Summary by Class")
            class_counts = {}
            for _, r in entries:
                for b in r.boxes:
                    n = r.names[int(b.cls[0])]
                    class_counts[n] = class_counts.get(n, 0) + 1
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                self.create_summary_card(page, class_name, count)

        self._divider(page)

        self._section_header(page, "🔍 Detailed Detections")
        for i, (name, r) in enumerate(entries, 1):
            self.create_image_card(page, i, name, r)

        self._refresh_results(page)

    def create_summary_card(self, page, class_name, count):
        """Create a summary card for each class"""
        card, sizer = self._make_card(page, 'mode_video', orient=wx.HORIZONTAL)
        sizer.Add(static_text(card, class_name.capitalize(), 12, True), 1,
                  wx.ALIGN_CENTER_VERTICAL | wx.ALL, 10)
        badge = static_text(card, f" {count} ", 12, True, '#ffffff')
        badge.SetBackgroundColour(C('mode_video'))
        sizer.Add(badge, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 10)
        page.results_sizer.Add(card, 0, wx.EXPAND | wx.ALL, 3)

    def create_image_card(self, page, index, name, result):
        """Per-image analysis card used by folder mode"""
        boxes = result.boxes
        card, sizer = self._make_card(page, 'mode_folder')

        head = wx.BoxSizer(wx.HORIZONTAL)
        head.Add(static_text(card, f"#{index}", 11, True, 'mode_folder'), 0,
                 wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        short = name if len(name) <= 26 else name[:23] + "..."
        head.Add(static_text(card, short, 12, True), 1, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 6)
        badge = static_text(card, f" {len(boxes)} ", 12, True, '#ffffff')
        badge.SetBackgroundColour(C('mode_folder'))
        head.Add(badge, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        sizer.Add(head, 0, wx.EXPAND | wx.TOP, 10)

        if not boxes:
            sizer.Add(static_text(card, "No objects detected", 10, False, 'fg_dim'),
                      0, wx.LEFT | wx.BOTTOM | wx.TOP, 12)
            page.results_sizer.Add(card, 0, wx.EXPAND | wx.ALL, 3)
            return

        confs = [float(b.conf[0]) for b in boxes]
        avg = sum(confs) / len(confs)
        counts = {}
        for b in boxes:
            n = result.names[int(b.cls[0])]
            counts[n] = counts.get(n, 0) + 1
        cls_text = ", ".join(f"{n.capitalize()} ×{c}" for n, c in
                             sorted(counts.items(), key=lambda x: x[1], reverse=True))

        conf_color = ('success' if avg > 0.7 else 'warning' if avg > 0.4 else 'error')
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(static_text(card, "Confidence:", 10, False, 'fg_dim'), 0,
                wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        gauge = wx.Gauge(card, range=1000, size=(100, 10))
        gauge.SetValue(int(max(0.0, min(1.0, avg)) * 1000))
        row.Add(gauge, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 6)
        row.Add(static_text(card, f"avg {avg * 100:.1f}% · max {max(confs) * 100:.1f}%",
                            10, True, conf_color), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 6)
        sizer.Add(row, 0, wx.TOP, 4)

        cls_st = static_text(card, cls_text, 10, False, 'fg_dim')
        cls_st.Wrap(320)
        sizer.Add(cls_st, 0, wx.LEFT | wx.RIGHT | wx.TOP | wx.BOTTOM, 10)

        page.results_sizer.Add(card, 0, wx.EXPAND | wx.ALL, 3)

    def create_detection_card(self, page, index, class_name, conf, x1, y1, x2, y2):
        """Create a detection card for individual detection"""
        card, sizer = self._make_card(page, 'mode_webcam')

        header = wx.BoxSizer(wx.HORIZONTAL)
        header.Add(static_text(card, f"#{index}", 11, True, 'mode_webcam'), 0,
                   wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        header.Add(static_text(card, class_name.capitalize(), 12, True), 1,
                   wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 6)
        sizer.Add(header, 0, wx.EXPAND | wx.TOP, 10)

        conf_pct = max(0.0, min(1.0, conf))
        conf_color = ('success' if conf_pct > 0.7 else 'warning' if conf_pct > 0.4 else 'error')

        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(static_text(card, "Confidence:", 10, False, 'fg_dim'), 0,
                wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        gauge = wx.Gauge(card, range=1000, size=(120, 10))
        gauge.SetValue(int(conf_pct * 1000))
        row.Add(gauge, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 6)
        row.Add(static_text(card, f"{conf_pct * 100:.1f}%", 10, True, conf_color), 0,
                wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 6)
        sizer.Add(row, 0, wx.TOP, 4)

        box_st = static_text(card, f"Box: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})",
                             10, False, 'fg_dim')
        sizer.Add(box_st, 0, wx.LEFT | wx.RIGHT | wx.TOP | wx.BOTTOM, 10)

        page.results_sizer.Add(card, 0, wx.EXPAND | wx.ALL, 3)

    # ------------------------------------------------------------------
    # Save / Clear
    # ------------------------------------------------------------------
    def on_save_result(self, evt):
        self.save_result()

    def save_result(self):
        """Save detection result (image, last video frame, or the 2x2 grid)"""
        page = self.active_page
        if page is None or page.last_result is None:
            wx.MessageBox("No result to save!", "Warning", wx.OK | wx.ICON_WARNING)
            return

        dlg = wx.FileDialog(
            self, message="Save Result", defaultFile="result.jpg",
            wildcard="JPEG (*.jpg)|*.jpg|PNG (*.png)|*.png|All files (*.*)|*.*",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() != wx.ID_OK:
            dlg.Destroy()
            return
        file_path = dlg.GetPath()
        dlg.Destroy()

        try:
            cv2.imwrite(file_path, cv2.cvtColor(page.last_result, cv2.COLOR_RGB2BGR))
            wx.MessageBox(f"Result saved to:\n{file_path}", "Success",
                          wx.OK | wx.ICON_INFORMATION)
            self.update_status(f"✓ Result saved: {Path(file_path).name}", COLORS['success'])
        except Exception as e:
            wx.MessageBox(f"Failed to save result:\n{str(e)}", "Error", wx.OK | wx.ICON_ERROR)

    def on_clear_display(self, evt):
        self.clear_display()

    def clear_display(self):
        """Clear display and results of the active page"""
        page = self.active_page
        if self.is_processing:
            self.paused = False
            self.stop_video = True

        page.display.show_placeholder()
        self.show_no_results(page)

        page.last_result = None
        if page.mode == 'folder':
            self.folder_path = None
            self.folder_images = []
            page.shuffle_btn.Enable(False)
        else:
            self.current_file = None
        self._set_file_label(page, "No file selected", 'error')
        if page.save_btn:
            page.save_btn.Enable(False)
        self.update_status("Display cleared", COLORS['fg_dim'])

    # ------------------------------------------------------------------
    def on_close(self, evt):
        self.stop_video = True
        self.pulse_timer.Stop()
        if self._splash is not None:
            self._splash.close()
        self.Destroy()


class YOLOApp(wx.App):
    def OnInit(self):
        self.frame = None
        self.splash = SplashFrame()
        self.splash.Show()
        self.splash.Update()
        self._create_timer = wx.CallLater(80, self._create_main)
        return True

    def _create_main(self):
        # Built (hidden) while the splash animates; shown once the model is ready
        self.frame = YOLODetectorFrame()
        self.SetTopWindow(self.frame)
        self.frame.begin_startup(self.splash)


def main():
    """Main application entry point"""
    app = YOLOApp(False)
    app.MainLoop()


if __name__ == "__main__":
    main()