"""Command-line version of the web app's image upload.

usage: python handlePNG.py IMAGE "text written in the image" STYLE_NUMBER
"""
import sys

from autotracecenter import tracetovector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    image, text, style = sys.argv[1], sys.argv[2], int(sys.argv[3])
    tracetovector(image, text, path_to_npy='./model/style/', name=f'style-{style}-',
                  preview=f'static/style-{style}-preview.png')
    print(f"Saved style {style}; preview at static/style-{style}-preview.png")
