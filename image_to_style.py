"""Save an image of handwriting as a numbered style in model/style/ (see hwlite/trace.py for how)."""
import numpy as np

from hwlite import Model
from hwlite.quality import best_style_from_image

WEIGHTS = 'docs/model/weights.npz'


def tracetovector(image_path=None, input_string=None, path_to_npy=None, name=None, threshold=None, preview=None):
    """Trace an image and save {path_to_npy}{name}strokes.npy / chars.npy.

    input_string must be the text written in the image. Several reconstructions
    are tried and the one the model writes most legibly from is kept.
    Returns (normalized stroke array, legibility score).
    """
    from StrokestoNPY import save_style
    strokes, score, _ = best_style_from_image(Model(np.load(WEIGHTS)), image_path, input_string, threshold)
    save_style(strokes, input_string, path_to_npy, name, preview=preview)
    return strokes, score
