"""Stroke format helpers (NumPy only).

A handwriting sample is an (n, 3) array of (dx, dy, pen_up): the pen's movement
since the previous point, and 1 where the pen is lifted after the point. The
scale is set so that the median movement between points is 1.
"""
import numpy as np

alphabet = [
    '\x00', ' ', '!', '"', '#', "'", '(', ')', ',', '-', '.',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
    '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z'
]
_alpha_to_num = {c: i for i, c in enumerate(alphabet)}

MAX_PRIME_POINTS = 1200   # longest sample the model is primed with
MAX_LINE_CHARS = 75


def encode_ascii(text):
    """Text -> array of character indices, terminated by 0."""
    return np.array([_alpha_to_num.get(c, 0) for c in text] + [0])


def check_text(lines):
    """Raise ValueError if any line is too long or uses characters the model never learned."""
    valid = set(alphabet) - {'\x00'}
    for number, line in enumerate(lines, 1):
        if len(line) > MAX_LINE_CHARS:
            raise ValueError(f"Line {number} has {len(line)} characters; the limit is {MAX_LINE_CHARS}.")
        for char in line:
            if char not in valid:
                raise ValueError(f"Invalid character {char!r} in line {number}. "
                                 f"The model can only write: {''.join(sorted(valid))}")


def align(coords):
    """Remove the overall slope of the baseline."""
    coords = np.copy(coords)
    x, y = coords[:, 0].reshape(-1, 1), coords[:, 1].reshape(-1, 1)
    x = np.concatenate([np.ones([x.shape[0], 1]), x], axis=1)
    offset, slope = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y).squeeze()
    theta = np.arctan(slope)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coords[:, :2] = np.dot(coords[:, :2], rotation) - offset
    return coords


_SAVGOL = np.array([-2, 3, 6, 7, 6, 3, -2]) / 21.0   # window 7, cubic


def _savgol(x):
    """Savitzky-Golay smoothing, same as scipy's savgol_filter(x, 7, 3, mode='nearest')."""
    return np.convolve(np.pad(x, 3, mode='edge'), _SAVGOL, mode='valid')


def denoise(coords):
    """Smooth each stroke separately."""
    pieces = np.split(coords, np.where(coords[:, 2] == 1)[0] + 1, axis=0)
    out = []
    for stroke in pieces:
        if len(stroke) != 0:
            out.append(np.column_stack([_savgol(stroke[:, 0]), _savgol(stroke[:, 1]), stroke[:, 2]]))
    return np.vstack(out)


def coords_to_offsets(coords):
    offsets = np.concatenate([coords[1:, :2] - coords[:-1, :2], coords[1:, 2:3]], axis=1)
    return np.concatenate([np.array([[0, 0, 1]]), offsets], axis=0)


def offsets_to_coords(offsets):
    return np.concatenate([np.cumsum(offsets[:, :2], axis=0), offsets[:, 2:3]], axis=1)


def normalize(offsets):
    offsets = np.copy(offsets)
    offsets[:, :2] /= np.median(np.linalg.norm(offsets[:, :2], axis=1))
    return offsets


def strokes_to_style(data):
    """Pen strokes -> the normalized (dx, dy, pen_up) array the model is primed with.

    data: list of {"x": [...], "y": [...]} strokes in writing order, y growing
    downward as on a canvas about 200 units tall.
    """
    rows = []
    for stroke in data:
        x = np.asarray(stroke["x"], dtype=np.float32)
        y = np.asarray(stroke["y"], dtype=np.float32)
        if len(x) == 0:
            continue
        pen = np.zeros_like(x)
        pen[-1] = 1
        rows.append(np.column_stack((x, y, pen)))
    if not rows or sum(len(r) for r in rows) < 2:
        raise ValueError("No strokes to process.")
    strokes = np.vstack(rows).astype(np.float32)
    strokes[:, 1] = 200 - strokes[:, 1]

    strokes = denoise(align(strokes))
    style = normalize(coords_to_offsets(strokes)).astype(np.float32)[:-1]
    if len(style) > MAX_PRIME_POINTS:
        raise ValueError(f"Sample has {len(style)} points; the model can be primed with at most "
                         f"{MAX_PRIME_POINTS}. Use a shorter sample (a few words).")
    return style
