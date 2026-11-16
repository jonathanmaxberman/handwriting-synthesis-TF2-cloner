"""Glue between the web page (worker.js) and hwlite. Runs inside Pyodide, in the browser."""
import io
import json
import re

import numpy as np

from hwlite import Writer
from hwlite.quality import legibility, verdict
from hwlite.strokes import check_text, offsets_to_coords, strokes_to_style

writer = None
rng = np.random.default_rng()


def init():
    global writer
    writer = Writer.from_files('/data/weights.npz', '/data/styles.npz')
    return json.dumps(sorted(writer.styles, key=int))


def generate(text, style, bias, color, justify, progress):
    lines, bold = [], []
    for raw in text.replace('\r', '').strip('\n').split('\n'):
        match = re.fullmatch(r'\s*\*\*(.+?)\*\*\s*', raw)
        lines.append((match.group(1) if match else raw).strip())
        bold.append(bool(match))
    if not any(lines):
        raise ValueError("Type some text to write.")
    svg, _ = writer.write(lines, style=None if style == 'none' else style, bias=float(bias), color=color,
                          justify=justify, bold=bold, rng=rng, progress=progress)
    return svg


def _sample_svg(strokes):
    """A small SVG picture of a style sample, to show what was captured."""
    coords = offsets_to_coords(np.vstack([strokes, [[0, 0, 1]]]))
    coords[:, 1] *= -1
    coords[:, :2] -= coords[:, :2].min(axis=0) - 4
    width, height = coords[:, 0].max() + 4, coords[:, 1].max() + 4
    path, pen_up = [], True
    for x, y, eos in coords:
        path.append(f"{'M' if pen_up else 'L'}{x:.1f},{y:.1f}")
        pen_up = eos == 1.0
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}">'
            f'<path d="{" ".join(path)}" fill="none" stroke="currentColor" stroke-width="1" '
            f'stroke-linecap="round" stroke-linejoin="round"/></svg>')


def _keep(strokes, text, score):
    # the style lives only in this page's memory: it is never uploaded or saved
    writer.add_style('mine', strokes, text)
    rating, message = verdict(score)
    return json.dumps({'rating': rating, 'score': round(score, 2), 'message': message,
                       'points': int(len(strokes)), 'sample': _sample_svg(strokes)})


def _check_sample_text(text):
    text = text.strip()
    if not text:
        raise ValueError("Type the words that the sample shows.")
    check_text([text])
    return text


def capture(strokes_json, text, progress):
    text = _check_sample_text(text)
    strokes = strokes_to_style(json.loads(strokes_json))
    if len(strokes) < 30:
        raise ValueError("That sample is too short. Write a few words.")
    return _keep(strokes, text, legibility(writer.model, strokes, text, rng=rng, progress=progress))


def trace(image_bytes, text, progress):
    from hwlite.quality import best_style_from_image
    text = _check_sample_text(text)
    strokes, score, _ = best_style_from_image(writer.model, io.BytesIO(bytes(image_bytes)), text, rng=rng,
                                              progress=progress)
    return _keep(strokes, text, score)
