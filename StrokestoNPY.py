import numpy as np

from handwriting_synthesis.drawing.operations import align, denoise, coords_to_offsets, normalize

# The model's priming buffer (Hand._sample) holds at most this many stroke points.
MAX_PRIME_POINTS = 1200


def strokes_to_style(data):
    """Turn strokes into the normalized (dx, dy, pen_up) array the model is primed with.

    data: list of {"x": [...], "y": [...]} strokes in writing order, y growing
    downward as on a canvas.
    """
    all_strokes = []
    for stroke_data in data:
        x = np.array(stroke_data["x"], dtype=np.float32)
        y = np.array(stroke_data["y"], dtype=np.float32)
        if len(x) == 0:
            continue

        # pen state: 0 = pen down, 1 = pen lifted after this point
        pen_state = np.zeros_like(x, dtype=np.float32)
        pen_state[-1] = 1

        all_strokes.extend(np.column_stack((x, y, pen_state)).tolist())

    if len(all_strokes) < 2:
        raise ValueError("No strokes to process.")
    strokes = np.array(all_strokes, dtype=np.float32)

    canvas_height = 200
    strokes[:, 1] = canvas_height - strokes[:, 1]

    strokes = align(strokes)
    strokes = denoise(strokes)
    offsets = coords_to_offsets(strokes)
    normalized_strokes = normalize(offsets).astype(np.float32)
    return normalized_strokes[:-1]


def save_style(normalized_strokes, priming_sequence, path, name, preview=None):
    """Save a style as {path}{name}strokes.npy and {path}{name}chars.npy (plus an optional preview image)."""
    if len(normalized_strokes) > MAX_PRIME_POINTS:
        raise ValueError(
            f"Sample has {len(normalized_strokes)} points; the model can be primed with at most "
            f"{MAX_PRIME_POINTS}. Use a shorter sample (a few words)."
        )
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
