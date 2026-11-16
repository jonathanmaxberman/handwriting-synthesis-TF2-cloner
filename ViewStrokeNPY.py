import sys

import numpy as np
from matplotlib import pyplot


def plot_stroke(stroke, save_name=None):
    """Plot a normalized (dx, dy, pen_up) stroke array; show it, or save it if save_name is given."""
    f, ax = pyplot.subplots()

    x = np.cumsum(stroke[:, 0])
    y = np.cumsum(stroke[:, 1])
    size_x = x.max() - x.min() + 1.0
    size_y = y.max() - y.min() + 1.0

    f.set_size_inches(5.0 * size_x / size_y, 5.0)

    # each pen lift ends a stroke; the last stroke runs to the end of the array
    cuts = list(np.where(stroke[:, 2] == 1)[0]) + [len(stroke) - 1]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value + 1], y[start:cut_value + 1], "k-", linewidth=3)
        start = cut_value + 1

    ax.axis("off")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(save_name, bbox_inches="tight", pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    pyplot.close()


if __name__ == '__main__':
    # usage: python ViewStrokeNPY.py model/style/style-16-strokes.npy
    file_path = sys.argv[1] if len(sys.argv) > 1 else './model/style/style-16-strokes.npy'
    stroke_data = np.load(file_path, allow_pickle=True)
    print(stroke_data, f'Size:{stroke_data.shape}')
    plot_stroke(stroke_data)
