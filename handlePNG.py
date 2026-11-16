"""Turn an image of handwriting into a numbered style, from the command line.

usage: python handlePNG.py IMAGE "text written in the image" STYLE_NUMBER

Needs only the packages in requirements.txt (no TensorFlow).
"""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    image, text, style = sys.argv[1], sys.argv[2], int(sys.argv[3])

    from image_to_style import tracetovector
    from hwlite.quality import verdict

    os.makedirs('static', exist_ok=True)
    _, score = tracetovector(image, text, path_to_npy='./model/style/', name=f'style-{style}-',
                             preview=f'static/style-{style}-preview.png')
    rating, message = verdict(score)
    print(f"Saved style {style}; preview at static/style-{style}-preview.png")
    print(f"Quality check: {rating} ({score:.2f}). {message}")
