"""Regenerate the example images shown on the demo page (docs/img/mode-*.png).

Uses hwlite only. docs/img/lincoln-sample.png is a public-domain scan of
Abraham Lincoln's handwriting reading "As I would not be a".
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hwlite import Writer
from hwlite.quality import best_style_from_image, legibility
from hwlite.strokes import align, denoise, offsets_to_coords

W = 1000


def font(size):
    for name in ("/System/Library/Fonts/Helvetica.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def ink(offsets, height, color=(20, 20, 20), width=2, ss=3, pad=14):
    """Draw one stroke array as an image `height` pixels tall."""
    c = denoise(offsets_to_coords(np.vstack([offsets, [[0, 0, 1]]])))
    c[:, :2] = align(c[:, :2])
    c[:, 1] *= -1
    c[:, :2] -= c[:, :2].min(axis=0)
    scale = (height - 2 * pad) / max(c[:, 1].max(), 1e-6)
    scale = min(scale, (W - 2 * pad) / max(c[:, 0].max(), 1e-6))
    c[:, :2] = c[:, :2] * scale * ss + pad * ss
    im = Image.new('RGB', (int(c[:, 0].max() + pad * ss), height * ss), 'white')
    d = ImageDraw.Draw(im)
    start = 0
    for cut in list(np.flatnonzero(c[:, 2] == 1)) + [len(c) - 1]:
        pts = [tuple(p) for p in c[start:cut + 1, :2]]
        if len(pts) > 1:
            d.line(pts, fill=color, width=width * ss, joint='curve')
        start = cut + 1
    return im.resize((im.width // ss, im.height // ss), Image.LANCZOS)


def sheet(rows, path):
    """rows: list of (caption or None, image). Stack them on a white sheet W pixels wide."""
    f = font(15)
    total = sum(im.height + (26 if cap else 0) + 8 for cap, im in rows) + 8
    out = Image.new('RGB', (W, total), 'white')
    d = ImageDraw.Draw(out)
    y = 8
    for cap, im in rows:
        if cap:
            d.text((14, y + 4), cap, fill=(110, 110, 110), font=f)
            y += 26
        if im.width > W:
            im = im.resize((W, int(im.height * W / im.width)), Image.LANCZOS)
        out.paste(im, (0, y))
        y += im.height + 8
    out.save(path, optimize=True)
    print(path, out.size)


if __name__ == '__main__':
    writer = Writer.from_files('docs/model/weights.npz', 'docs/model/styles.npz')
    rng = np.random.default_rng(7)

    def write(lines, style, bias=0.9):
        _, strokes = writer.write(lines, style=style, bias=bias, rng=rng)
        return strokes

    # 1. built-in styles
    line = "The quick brown fox jumps over the lazy dog"
    sheet([(f"built-in style {s}", ink(write([line], s)[0], 70)) for s in ("7", "3", "9", "12")],
          'docs/img/mode-write.png')

    # 2. a pen-recorded sample -> new text
    sample, text = writer.styles["0"]
    new = write(["A handwriting style can be picked up", "from just a few words"], "0")
    sheet([(f'pen-recorded sample: "{text}"', ink(sample, 80, color=(31, 78, 140)))] +
          [("new text in that style" if i == 0 else None, ink(s, 70)) for i, s in enumerate(new)],
          'docs/img/mode-draw.png')

    # 3. a photo -> reconstructed strokes -> new text
    scan = Image.open('docs/img/lincoln-sample.png').convert('RGB')
    scan = scan.resize((int(scan.width * 110 / scan.height), 110), Image.LANCZOS)
    strokes, score, _ = best_style_from_image(writer.model, 'docs/img/lincoln-sample.png', "As I would not be a", rng=rng)
    print("lincoln score", round(score, 2))
    best = None
    for _ in range(4):   # the model's output varies from run to run; show a representative legible one
        new = write(["Four score and seven years ago", "our fathers brought forth"], (strokes, "As I would not be a"))
        s = np.mean([writer.model.nll(n, "x" * 20) - writer.model.nll(n, t) for n, t in
                     zip(new, ["Four score and seven years ago", "our fathers brought forth"])])
        if best is None or s > best[0]:
            best = (s, new)
    sheet([("scan of handwriting", scan), ("pen strokes reconstructed from the scan", ink(strokes, 80, color=(31, 78, 140)))] +
          [("new text in that style" if i == 0 else None, ink(s, 70)) for i, s in enumerate(best[1])],
          'docs/img/mode-photo.png')
