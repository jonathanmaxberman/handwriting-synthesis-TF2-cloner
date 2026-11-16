![](img/banner.svg)

# Handwriting Synthesis (TFv2)

Original repo migrated to tensorflow v2 with better structure

---

Implementation of the handwriting synthesis experiments in the paper <a href="https://arxiv.org/abs/1308.0850">
Generating Sequences with Recurrent Neural Networks</a> by Alex Graves. The implementation closely follows the original
paper, with a few slight deviations, and the generated samples are of similar quality to those presented in the paper.

## Installation

Needs Python 3.10-3.12 (3.12 is what this was last verified on, with TensorFlow 2.21).

```shell
git clone https://github.com/jonathanmaxberman/handwriting-synthesis-TF2-cloner
cd handwriting-synthesis-TF2-cloner
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Web app (with handwriting cloning)

```shell
python app.py
```

Then open http://127.0.0.1:5001. From the page you can:

- type text and have it written in any of the saved styles;
- **draw** a few words in the box, type what you wrote as the priming sequence, and press
  "Capture Strokes" to save your own handwriting as a new style;
- **upload an image** of handwriting (one line of a few words, cropped close; a phone photo is
  fine), type the words it contains, and press "Trace Image".

Both report a quality check: the model is primed with the new style, asked to write a few test
lines, and then scored on whether what it wrote is legible (see `style_quality.py`). "good" is
about 0.3 and up; real pen recordings score 0.4-0.6.

The same image tracing is available from the command line:

```shell
python handlePNG.py sample.png "the words in the image" 17
```

### How an image becomes a style

The model is primed with *online* handwriting: pen positions in the order they were written,
sampled at a tablet's rate, with pen lifts marked. An image has none of that, so
`image_to_style.py` reconstructs it:

1. clean the image (transparency, light-on-dark, uneven lighting, automatic threshold, specks);
2. thin the ink to its centerline and turn it into a graph of lines and junctions;
3. walk each connected piece of ink the way a pen would: straight on through crossings, doubling
   back along the ink where a hand does;
4. order the strokes (words left to right, dots and crosses after their word);
5. resample with a human-like speed profile (slower in curves and at stroke ends), which matters
   a lot: the model treats evenly spaced points as unnatural;
6. try a few reconstructions (more or less doubling back, faster or slower pen) and keep the one
   the model writes most legibly from.

Measured by rendering the 13 shipped styles to images and tracing them back: 12 of 13 gave a
usable style from clean images and 9 of 13 from simulated phone photos. Those are synthetic
images, not real scans. Very loopy cursive is the known weak spot.

### Responsible use

This imitates the general look of a handwriting; it does not reproduce signatures and is not a
forgery tool. Only clone handwriting that is yours or that you have permission to use, and do
not present generated writing as having been written by someone else.

The demo script still works too:

```shell
python main.py
```

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
