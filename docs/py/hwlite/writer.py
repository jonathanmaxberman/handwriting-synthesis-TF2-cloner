"""Write text as handwriting and render it as SVG (NumPy only)."""
import io

import numpy as np

from .model import Model
from .strokes import align, check_text, denoise, offsets_to_coords


def to_svg(strokes, lines, color='black', widths=None, justify='left', line_height=60, view_width=1000):
    """Render generated stroke arrays (one per line) as an SVG document string."""
    widths = widths or [2] * len(lines)
    view_height = line_height * (len(lines) + 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_width} {view_height}">',
             f'<rect x="0" y="0" width="{view_width}" height="{view_height}" fill="white"/>']
    y0 = 3 * line_height / 4
    for offsets, line, width in zip(strokes, lines, widths):
        if not line or len(offsets) < 2:
            y0 += line_height
            continue
        offsets = offsets.copy()
        offsets[:, :2] *= 1.5
        coords = denoise(offsets_to_coords(offsets))
        coords[:, :2] = align(coords[:, :2])
        coords[:, 1] *= -1
        coords[:, 0] -= coords[:, 0].min()
        coords[:, 1] += y0 - coords[:, 1].min()
        ink_width = coords[:, 0].max()
        margin = 10
        if justify == 'left':
            coords[:, 0] += margin
        elif justify == 'right':
            coords[:, 0] += view_width - ink_width - margin
        else:
            coords[:, 0] += (view_width - ink_width) / 2

        path, pen_up = [], True
        for x, y, eos in coords:
            path.append(f"{'M' if pen_up else 'L'}{x:.1f},{y:.1f}")
            pen_up = eos == 1.0
        parts.append(f'<path d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="{width}" '
                     f'stroke-linecap="round" stroke-linejoin="round"/>')
        y0 += line_height
    parts.append('</svg>')
    return '\n'.join(parts)


class Writer:
    """The model plus a set of named styles."""

    def __init__(self, weights, styles=None):
        """weights: opened .npz of model weights. styles: opened .npz with '<name>-strokes' / '<name>-chars' entries."""
        self.model = Model(weights)
        self.styles = {}
        if styles is not None:
            for key in styles:
                if key.endswith('-strokes'):
                    name = key[:-len('-strokes')]
                    self.styles[name] = (np.asarray(styles[key], dtype=np.float32), str(styles[name + '-chars']))

    @classmethod
    def from_files(cls, weights_path, styles_path=None):
        return cls(np.load(weights_path), np.load(styles_path) if styles_path else None)

    def add_style(self, name, strokes, text):
        self.styles[str(name)] = (np.asarray(strokes, dtype=np.float32), text)

    def write(self, lines, style=None, bias=0.75, color='black', justify='left', bold=None, rng=None, progress=None):
        """Write lines of text; returns (svg string, list of stroke arrays).

        style: a style name, or a (strokes, text) pair, or None for the model's own hand.
        bold: optional list of booleans, one per line.
        """
        check_text(lines)
        prime = self.styles[str(style)] if isinstance(style, (str, int)) else style
        todo = [i for i, line in enumerate(lines) if line]
        strokes = [np.zeros((0, 3), dtype=np.float32) for _ in lines]
        if todo:
            written = self.model.write([lines[i] for i in todo], biases=[bias] * len(todo),
                                       primes=[prime] * len(todo) if prime is not None else None,
                                       rng=rng, progress=progress)
            for i, s in zip(todo, written):
                if prime is not None:
                    # the first few samples are the tail end of the style sample's last stroke: drop them
                    lifts = np.flatnonzero(s[:6, 2] == 1)
                    if len(lifts):
                        s = s[lifts[-1] + 1:]
                strokes[i] = s
        widths = [2 if b else 1 for b in bold] if bold else [1] * len(lines)
        return to_svg(strokes, lines, color=color, widths=widths, justify=justify), strokes
