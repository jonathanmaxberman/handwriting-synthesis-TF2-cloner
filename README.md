![](img/banner.svg)

# Handwriting Synthesis (TFv2)

Original repo migrated to tensorflow v2 with better structure

---

Implementation of the handwriting synthesis experiments in the paper <a href="https://arxiv.org/abs/1308.0850">
Generating Sequences with Recurrent Neural Networks</a> by Alex Graves. The implementation closely follows the original
paper, with a few slight deviations, and the generated samples are of similar quality to those presented in the paper.

## Live demo

**https://jonathanmaxberman.github.io/handwriting-synthesis-TF2-cloner/**

The demo is a static page (the `docs/` folder, served by GitHub Pages). There is no server: the
model runs in the visitor's own browser, as Python compiled to WebAssembly (Pyodide), so text and
handwriting samples never leave their device and a captured style is gone when the tab closes.
It is a demonstration for informational purposes.

On the page you can:

- type text and have it written in any of the 13 built-in styles;
- **draw** a few words, type what you wrote, and have the model write in your handwriting. The
  model then tests itself in that style and reports whether its writing is legible.

Expect a download of about 25 MB on first use, and roughly 10 seconds to write a couple of lines
on a laptop (the quality test of a new style takes about 25 seconds).

### Cloning a style from a photo: included, switched off

The code can also pick up a style from a *photo* of handwriting (`hwlite/trace.py`). On the public
page that interface is deliberately switched off, as a small barrier against casual misuse such as
forged homework, notes or prescriptions. It is a speed bump, not a lock. To study it, run your own
copy and flip the switch:

1. in `docs/config.js` set `ENABLE_IMAGE_CLONING: true`;
2. serve the folder locally: `python -m http.server --directory docs 8000` and open
   http://localhost:8000.

The first use downloads about 45 MB of extra image tools, and tracing takes a few minutes in the
browser because five reconstructions are each tested. From the command line it takes about ten
seconds:

```shell
python handlePNG.py sample.png "the words in the image" 17
```

Use only handwriting that is yours or that you have permission to use.

## Installation (for running the code yourself)

Python 3.10+. The model runs in plain NumPy, so TensorFlow is **not** needed unless you want to
retrain it.

```shell
git clone https://github.com/jonathanmaxberman/handwriting-synthesis-TF2-cloner
cd handwriting-synthesis-TF2-cloner
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```python
from hwlite import Writer

writer = Writer.from_files('docs/model/weights.npz', 'docs/model/styles.npz')
svg, strokes = writer.write(["Hello there", "this is a test"], style="7", bias=0.85)
open('hello.svg', 'w').write(svg)
```

### Layout

| Path | What it is |
|---|---|
| `hwlite/` | The model and everything around it in NumPy: `model.py` (the network), `strokes.py` (stroke format), `writer.py` (text to SVG), `trace.py` (image to strokes), `quality.py` (legibility test) |
| `docs/` | The demo site. `docs/py/hwlite` is a copy of `hwlite/` made by `build_site.py`; `docs/model/` holds the weights and built-in styles |
| `handwriting_synthesis/` | The original TensorFlow implementation, kept for training (`pip install -r requirements-train.txt`) |
| `export_weights.py` | Converts a TensorFlow checkpoint to `docs/model/weights.npz` |
| `make_demo_images.py` | Regenerates the example images on the demo page |

`hwlite` was checked against the TensorFlow model: the likelihood it assigns to a handwriting
sample agrees to five decimal places.

### How an image becomes a style

The model is primed with *online* handwriting: pen positions in the order they were written,
sampled at a tablet's rate, with pen lifts marked. An image has none of that, so
`hwlite/trace.py` reconstructs it:

1. clean the image (transparency, light-on-dark, uneven lighting, automatic threshold, specks);
2. thin the ink to its centerline and turn it into a graph of lines and junctions;
3. walk each connected piece of ink the way a pen would: straight on through crossings, doubling
   back along the ink where a hand does;
4. order the strokes (words left to right, dots and crosses after their word);
5. resample with a human-like speed profile (slower in curves and at stroke ends), which matters
   a lot: the model treats evenly spaced points as unnatural;
6. try a few reconstructions (more or less doubling back, faster or slower pen) and keep the one
   the model writes most legibly from (`hwlite/quality.py`).

Measured by rendering the 13 built-in styles to images and tracing them back: 12 of 13 gave a
usable style from clean images and 9 of 13 from simulated phone photos. Those are synthetic
images, not real scans. Very loopy cursive is the known weak spot.

### Responsible use

This imitates the general look of a handwriting; it does not reproduce signatures and is not a
forgery tool. Only clone handwriting that is yours or that you have permission to use, and do
not present generated writing as having been written by someone else. The trained network comes
from the IAM On-Line Handwriting Database, which is provided for non-commercial research.

## Original TensorFlow version

Everything below describes the upstream TensorFlow code in `handwriting_synthesis/`, which still
works (`pip install -r requirements-train.txt`, then `python main.py`).

## Usage

```python
from handwriting_synthesis.hand import Hand

lines = [
    "Father time, I'm running late",
    "I'm winding down, I'm growing tired",
    "Seconds drift into the night",
    "The clock just ticks till my time expires",
]
biases = [.75 for _ in lines]
styles = [9 for _ in lines]
stroke_colors = ['red', 'green', 'black', 'blue']
stroke_widths = [1, 2, 1, 2]

hand = Hand()
hand.write(
    filename='img/usage_demo.svg',
    lines=lines,
    biases=biases,
    styles=styles,
    stroke_colors=stroke_colors,
    stroke_widths=stroke_widths
)
```

![](img/usage_demo.svg)

## Demonstrations

Below are a few hundred samples from the model, including some samples demonstrating the effect of priming and biasing
the model. Loosely speaking, biasing controls the neatness of the samples and priming controls the style of the samples.
The code for these demonstrations can be found in `main.py`.

### Demo #1:

The following samples were generated with a fixed style and fixed bias.

**Smash Mouth – All Star (<a href="https://www.azlyrics.com/lyrics/smashmouth/allstar.html">lyrics</a>)**
![](img/all_star.svg)

### Demo #2

The following samples were generated with varying style and fixed bias. Each verse is generated in a different style.

**Vanessa Carlton – A Thousand Miles (<a href="https://www.azlyrics.com/lyrics/vanessacarlton/athousandmiles.html">
lyrics</a>)**
![](img/downtown.svg)

### Demo #3

The following samples were generated with a fixed style and varying bias. Each verse has a lower bias than the previous,
with the last verse being unbiased.

**Leonard Cohen – Hallelujah (<a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ">lyrics</a>)**
![](img/give_up.svg)

## Training

A pretrained model is included, but if you'd like to train your own,
read [these instructions](model/README.md)

## Contribute

All Pull Requests and packaging  are welcome

You can take a look at the [issues](https://github.com/otuva/handwriting-synthesis/issues)

If there aren't any, you can see [TODO.md](TODO.md)

If you are having a problem or have a feature request, feel free to open issue

---

This project was intended to serve as a reference implementation for a research paper, but since the results are of
decent quality, it may be worthwile to make the project more broadly usable. I plan to continue focusing on the machine
learning side of things. That said, I'd welcome contributors who can:

- Package this, and otherwise make it look more like a usable software project and less like research code.
- Add support for more sophisticated drawing, animations, or anything else in this direction. Currently, the project
  only creates some simple svg files.
