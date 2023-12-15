![](img/banner.svg)
# Handwriting Synthesis
Implementation of the handwriting synthesis experiments in the paper <a href="https://arxiv.org/abs/1308.0850">Generating Sequences with Recurrent Neural Networks</a> by Alex Graves.  The implementation closely follows the original paper, with a few slight deviations, and the generated samples are of similar quality to those presented in the paper.

Web demo is available <a href="https://seanvasquez.com/handwriting-generation/">here</a>.

## Usage
```python
lines = [
    "Now this is a story all about how",
    "My life got flipped turned upside down",
    "And I'd like to take a minute, just sit right there",
    "I'll tell you how I became the prince of a town called Bel-Air",
]
biases = [.75 for i in lines]
styles = [9 for i in lines]
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

Currently, the `Hand` class must be imported from `demo.py`.  If someone would like to package this project to make it more usable, please [contribute](#contribute).

A pretrained model is included, but if you'd like to train your own, read <a href='https://github.com/sjvasquez/handwriting-synthesis/tree/master/data/raw'>these instructions</a>.





## Modifications
The file Markdown.py allows you to enter text with normal linebreaks instead of as individual lines. 

The file tracing.py will take a .png image and transcript of the text and attempt to generate a trace of the strokes to prime the model to adopt that style. This is a slightly adapted version of this project: https://github.com/amR0ssi/handwritten-character-skeleton, outputting the contours to a format that can by translated into a style. 

By default this are saved as style 16. Currently this is pretty hit or miss and you might need to try a few areas of text before you get workable copy.

The stroke styles in /styles/ are tensor in numpy format. The array structure is [x, y penup], where during a penstroke penup is 0, and at the stary and end it is 1. 
ViewStrokeNPY.py contains a function to plot, so you can see the file contents. This code is adapted from: https://github.com/swechhasingh/Handwriting-synthesis. 

The flask application app.py will run locally at localhost:5000. It will allow a few things. 

First it provides a way to enter your own handwriting for priming by writing in a Jsignature box, and providing transcript. By default these are saved as style 15.
Second it provides an easy mechanism to enter text and try various styles.
Third it provides the option to left-justify output. 

The "bias" setting controls how closely the output sticks to the model's probability distribbution. A high number (close to 1) will be more readable, but have little variation. A lower number will vary more, but may be less readable. 
There are 14 priming styles included, but you can make more by writing, or by tracing images. 




## Demonstrations
Below are a few hundred samples from the model, including some samples demonstrating the effect of priming and biasing the model.  Loosely speaking, biasing controls the neatness of the samples and priming controls the style of the samples. The code for these demonstrations can be found in `demo.py`.

### Demo #1:
The following samples were generated with a fixed style and fixed bias.

**Smash Mouth – All Star (<a href="https://www.azlyrics.com/lyrics/smashmouth/allstar.html">lyrics</a>)**
![](img/all_star.svg)

### Demo #2
The following samples were generated with varying style and fixed bias.  Each verse is generated in a different style.

**Vanessa Carlton – A Thousand Miles (<a href="https://www.azlyrics.com/lyrics/vanessacarlton/athousandmiles.html">lyrics</a>)**
![](img/downtown.svg)

### Demo #3
The following samples were generated with a fixed style and varying bias.  Each verse has a lower bias than the previous, with the last verse being unbiased.

**Leonard Cohen – Hallelujah (<a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ">lyrics</a>)**
![](img/give_up.svg)

## Contribute
This project was intended to serve as a reference implementation for a research paper, but since the results are of decent quality, it may be worthwile to make the project more broadly usable.  I plan to continue focusing on the machine learning side of things.  That said, I'd welcome contributors who can:

  - Package this, and otherwise make it look more like a usable software project and less like research code.
  - Add support for more sophisticated drawing, animations, or anything else in this direction.  Currently, the project only creates some simple svg files.
