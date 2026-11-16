"""Refresh the generated parts of the demo site in docs/.

- copies the hwlite package to docs/py/hwlite (GitHub Pages can only serve docs/)
- packs the built-in styles into docs/model/styles.npz

Run after changing hwlite/ or the built-in styles. The model weights
(docs/model/weights.npz) are produced separately by export_weights.py.
"""
import glob
import os
import shutil

import numpy as np

BUILT_IN_STYLES = range(13)   # style-0 .. style-12 shipped with the model

if __name__ == '__main__':
    os.makedirs('docs/py/hwlite', exist_ok=True)
    for path in glob.glob('hwlite/*.py'):
        shutil.copy(path, 'docs/py/hwlite/')

    styles = {}
    for n in BUILT_IN_STYLES:
        styles[f'{n}-strokes'] = np.load(f'model/style/style-{n}-strokes.npy').astype(np.float32)
        styles[f'{n}-chars'] = np.array(np.load(f'model/style/style-{n}-chars.npy').tobytes().decode('utf-8'))
    np.savez('docs/model/styles.npz', **styles)
    print(f"site refreshed: {len(glob.glob('docs/py/hwlite/*.py'))} python files, {len(BUILT_IN_STYLES)} styles")
