"""Check whether a handwriting sample actually works as a style, using the model itself.

A style "works" when the model, primed with it, goes on to write the text it
is asked for. That is tested directly: prime the model with the sample, have
it write a few short test lines, then ask the same model how well the strokes
it produced fit the requested text compared with a different text. Legible
writing of the right words scores about 0.4-0.6; a sample that derails the
model scores about 0.
"""
import numpy as np

from handwriting_synthesis import drawing

TEST_LINES = ["The quick brown fox", "over the lazy dog", "We will meet again"]
GOOD, POOR = 0.30, 0.15


def _nll(hand, strokes, text):
    """Average negative log-likelihood per pen sample of strokes given text (teacher forced)."""
    nn = hand.nn
    chars = drawing.encode_ascii(text)
    x, y = strokes[None, :-1], strokes[None, 1:]
    return float(nn.session.run(nn.loss, {nn.x: x, nn.y: y, nn.x_len: [x.shape[1]],
                                          nn.c: chars[None], nn.c_len: [len(chars)]}))


def legibility(hand, style_strokes, style_text, bias=0.8):
    """Score a style (normalized stroke array + the text it shows); higher is better."""
    wrong = TEST_LINES[1:] + TEST_LINES[:1]
    generated = hand._sample(TEST_LINES, biases=[bias] * len(TEST_LINES),
                             styles=[(style_strokes, style_text)] * len(TEST_LINES))
    scores = []
    for strokes, right_text, wrong_text in zip(generated, TEST_LINES, wrong):
        if len(strokes) < 20:
            scores.append(0.0)
            continue
        strokes = strokes.astype(np.float32)
        scores.append(_nll(hand, strokes, wrong_text) - _nll(hand, strokes, right_text))
    return float(np.mean(scores))


def verdict(score):
    if score >= GOOD:
        return "good", "The model writes legibly in this style."
    if score >= POOR:
        return "fair", "The model partly follows this style; results may be uneven. A different sample may work better."
    return "poor", ("The model could not write legibly from this sample. Try a different sample: a few clearly "
                    "written words, with the priming text matching them exactly.")


def best_style_from_image(hand, image_path, text, threshold=None):
    """Trace an image several ways and keep the reconstruction the model handles best.

    How far a pen doubles back along its own ink (print vs. cursive) and how
    fast the writer was cannot be read off an image, so a few plausible
    reconstructions are tried and scored. Returns (strokes, score, settings).
    """
    from image_to_style import load_ink_mask, strokes_from_ink, strokes_to_offsets

    ink = load_ink_mask(image_path, threshold)

    def candidate(retrace, density):
        strokes = strokes_to_offsets(strokes_from_ink(ink, text, density=density, retrace=retrace))
        return strokes, legibility(hand, strokes, text)

    tried = {}
    for retrace in (0.5, 0.2, 0.0):
        tried[(retrace, 1.0)] = candidate(retrace, 1.0)
    retrace = max(tried, key=lambda k: tried[k][1])[0]
    for density in (0.85, 1.2):
        tried[(retrace, density)] = candidate(retrace, density)

    settings = max(tried, key=lambda k: tried[k][1])
    strokes = tried[settings][0]
    # the winning score is flattered by having been picked as the maximum; measure it again
    score = legibility(hand, strokes, text)
    return strokes, score, {'retrace': settings[0], 'density': settings[1]}
