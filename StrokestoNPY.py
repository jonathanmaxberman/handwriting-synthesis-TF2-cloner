"""Save pen strokes as a numbered style in model/style/."""
import numpy as np

from hwlite.strokes import MAX_PRIME_POINTS, strokes_to_style  # noqa: F401


def save_style(normalized_strokes, priming_sequence, path, name, preview=None):
    """Save a style as {path}{name}strokes.npy and {path}{name}chars.npy (plus an optional preview image)."""
    np.save(f'{path}{name}strokes.npy', normalized_strokes)
    np.save(f'{path}{name}chars.npy', priming_sequence.encode('utf-8'))
    if preview:
        from ViewStrokeNPY import plot_stroke
        plot_stroke(normalized_strokes, save_name=preview)


def handle_draw_data(data, priming_sequence, path, name, preview=None):
    """Turn captured strokes into a priming style, save it, and return the normalized stroke array."""
    normalized_strokes = strokes_to_style(data)
    save_style(normalized_strokes, priming_sequence, path, name, preview=preview)
    return normalized_strokes
