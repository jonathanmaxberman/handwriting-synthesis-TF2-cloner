"""Export the trained TensorFlow checkpoint to a plain NumPy file for hwlite / the web demo.

Only needed again if the model is retrained. Requires TensorFlow.
usage: python export_weights.py [float16|float32]
"""
import sys

import numpy as np
import tensorflow as tf

NAMES = {
    'rnn/LSTMAttentionCell/lstm_cell/kernel': 'lstm1_kernel',
    'rnn/LSTMAttentionCell/lstm_cell/bias': 'lstm1_bias',
    'rnn/LSTMAttentionCell/lstm_cell_1/kernel': 'lstm2_kernel',
    'rnn/LSTMAttentionCell/lstm_cell_1/bias': 'lstm2_bias',
    'rnn/LSTMAttentionCell/lstm_cell_2/kernel': 'lstm3_kernel',
    'rnn/LSTMAttentionCell/lstm_cell_2/bias': 'lstm3_bias',
    'rnn/LSTMAttentionCell/attention/weights': 'attention_weights',
    'rnn/LSTMAttentionCell/attention/biases': 'attention_biases',
    'rnn/gmm/weights': 'gmm_weights',
    'rnn/gmm/biases': 'gmm_biases',
}

if __name__ == '__main__':
    dtype = sys.argv[1] if len(sys.argv) > 1 else 'float16'
    reader = tf.train.load_checkpoint(tf.train.latest_checkpoint('model/checkpoint'))
    weights = {short: reader.get_tensor(name).astype(dtype) for name, short in NAMES.items()}
    np.savez('docs/model/weights.npz', **weights)
    print({k: v.shape for k, v in weights.items()})
