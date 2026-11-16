"""Check whether a handwriting sample actually works as a style, using the model itself.

A style "works" when the model, primed with it, goes on to write the text it
is asked for. That is tested directly: prime the model with the sample, have
it write a few short test lines, then ask the same model how well the strokes
it produced fit the requested text compared with a different text. Legible
writing of the right words scores about 0.4-0.6; a sample that derails the
model scores about 0.
"""
import numpy as np

TEST_LINES = ["The quick brown fox", "over the lazy dog", "We will meet again"]
GOOD, POOR = 0.30, 0.15


def legibility(model, style_strokes, style_text, bias=0.8, rng=None, progress=None):
    """Score a style (normalized stroke array + the text it shows); higher is better."""
    wrong = TEST_LINES[1:] + TEST_LINES[:1]
    generated = model.write(TEST_LINES, biases=[bias] * len(TEST_LINES),
                            primes=[(style_strokes, style_text)] * len(TEST_LINES), rng=rng, progress=progress)
    scores = []
    for strokes, right_text, wrong_text in zip(generated, TEST_LINES, wrong):
        if len(strokes) < 20:
            scores.append(0.0)
            continue
        scores.append(model.nll(strokes, wrong_text) - model.nll(strokes, right_text))
    return float(np.mean(scores))


def verdict(score):
    if score >= GOOD:
        return "good", "The model writes legibly in this style."
    if score >= POOR:
        return "fair", "The model partly follows this style; results may be uneven. A different sample may work better."
    return "poor", ("The model could not write legibly from this sample. Try a different sample: a few clearly "
                    "written words, with the text matching them exactly.")


def best_style_from_image(model, image, text, threshold=None, rng=None, progress=None):
    """Trace an image several ways and keep the reconstruction the model handles best.

    How far a pen doubles back along its own ink (print vs. cursive) and how
    fast the writer was cannot be read off an image, so a few plausible
    reconstructions are tried and scored. Returns (strokes, score, settings).
    """
    from .trace import load_ink_mask, strokes_from_ink, strokes_to_offsets

    ink = load_ink_mask(image, threshold)
    plan = 6
    done = [0]

    def candidate(retrace, density):
        strokes = strokes_to_offsets(strokes_from_ink(ink, text, density=density, retrace=retrace))
        score = legibility(model, strokes, text, rng=rng,
                           progress=(lambda f: progress((done[0] + f) / plan)) if progress else None)
        done[0] += 1
        return strokes, score

    tried = {}
    for retrace in (0.5, 0.2, 0.0):
        tried[(retrace, 1.0)] = candidate(retrace, 1.0)
    retrace = max(tried, key=lambda k: tried[k][1])[0]
    for density in (0.85, 1.2):
        tried[(retrace, density)] = candidate(retrace, density)

    settings = max(tried, key=lambda k: tried[k][1])
    strokes = tried[settings][0]
    # the winning score is flattered by having been picked as the maximum; measure it again
    score = legibility(model, strokes, text, rng=rng,
                       progress=(lambda f: progress((done[0] + f) / plan)) if progress else None)
    return strokes, score, {'retrace': settings[0], 'density': settings[1]}
