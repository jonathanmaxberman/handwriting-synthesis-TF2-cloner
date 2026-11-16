import os

# The model is a TF1-style graph (tf.compat.v1 RNN cells / layers). Those APIs
# only exist under Keras 2, which modern TensorFlow ships as the separate
# `tf_keras` package. This must be set before tensorflow is first imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from .hand import Hand  # noqa: E402
