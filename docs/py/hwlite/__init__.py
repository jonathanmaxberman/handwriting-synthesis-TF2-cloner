"""Handwriting synthesis that needs only NumPy.

The same trained network as the TensorFlow code in handwriting_synthesis/,
re-implemented in plain NumPy so that it runs anywhere - including inside a web
page (via Pyodide), which is how the demo site in docs/ works.
"""
from .model import Model  # noqa: F401
from .writer import Writer  # noqa: F401
