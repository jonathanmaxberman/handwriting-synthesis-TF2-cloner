"""The handwriting network (Graves 2013) as plain NumPy.

Three LSTM layers with a soft attention window over the text, and a mixture
density output: at every step the network predicts a distribution over the
pen's next movement, and writing is done by sampling from it step by step.
This mirrors handwriting_synthesis/rnn (TensorFlow) exactly and loads the same
trained weights, exported by export_weights.py.
"""
import numpy as np

from .strokes import alphabet, encode_ascii

LSTM, ATTN_K, OUT_K = 400, 10, 20
N_CHARS = len(alphabet)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.logaddexp(0.0, x)


class Model:
    def __init__(self, weights):
        """weights: mapping of name -> array (an .npz file opened with numpy.load works)."""
        w = {k: np.asarray(weights[k], dtype=np.float32) for k in weights}
        self.lstm = [(w[f'lstm{i}_kernel'], w[f'lstm{i}_bias']) for i in (1, 2, 3)]
        self.attn = (w['attention_weights'], w['attention_biases'])
        self.gmm = (w['gmm_weights'], w['gmm_biases'])

    # ------------------------------------------------------------------ one time step

    @staticmethod
    def zero_state(batch, n_text):
        z = lambda n: np.zeros((batch, n), dtype=np.float32)
        return dict(h1=z(LSTM), c1=z(LSTM), h2=z(LSTM), c2=z(LSTM), h3=z(LSTM), c3=z(LSTM),
                    kappa=z(ATTN_K), w=z(N_CHARS), phi=z(n_text))

    @staticmethod
    def _lstm(params, x, c, h):
        kernel, bias = params
        i, j, f, o = np.split(np.concatenate([x, h], axis=1) @ kernel + bias, 4, axis=1)
        c = _sigmoid(f + 1.0) * c + _sigmoid(i) * np.tanh(j)
        return c, _sigmoid(o) * np.tanh(c)

    def step(self, x, state, text_onehot, text_mask):
        """Advance by one pen sample x (batch, 3)."""
        c1, h1 = self._lstm(self.lstm[0], np.concatenate([state['w'], x], axis=1), state['c1'], state['h1'])

        # attention: a mixture of Gaussians over character positions that slides along the text
        p = _softplus(np.concatenate([state['w'], x, h1], axis=1) @ self.attn[0] + self.attn[1])
        alpha, beta, kappa = np.split(p, 3, axis=1)
        kappa = state['kappa'] + kappa / 25.0
        beta = np.maximum(beta, 0.01)
        u = np.arange(text_onehot.shape[1], dtype=np.float32)[None, None, :]
        phi = (alpha[:, :, None] * np.exp(-np.square(kappa[:, :, None] - u) / beta[:, :, None])).sum(axis=1)
        w = ((phi * text_mask)[:, :, None] * text_onehot).sum(axis=1)

        c2, h2 = self._lstm(self.lstm[1], np.concatenate([x, h1, w], axis=1), state['c2'], state['h2'])
        c3, h3 = self._lstm(self.lstm[2], np.concatenate([x, h2, w], axis=1), state['c3'], state['h3'])
        return dict(h1=h1, c1=c1, h2=h2, c2=c2, h3=h3, c3=c3, kappa=kappa, w=w, phi=phi)

    def _mixture(self, h3, bias=None):
        """Parameters of the predicted distribution over the next pen sample."""
        z = h3 @ self.gmm[0] + self.gmm[1]
        pis, sigmas, rhos, mus, es = np.split(z, [OUT_K, 3 * OUT_K, 4 * OUT_K, 6 * OUT_K], axis=1)
        if bias is not None:
            # bias sharpens the distribution: neater, less varied writing
            pis = pis * (1 + bias[:, None])
            sigmas = sigmas - bias[:, None]
        pis = np.exp(pis - pis.max(axis=1, keepdims=True))
        pis /= pis.sum(axis=1, keepdims=True)
        sigmas = np.maximum(np.exp(sigmas), 1e-4)
        rhos = np.clip(np.tanh(rhos), 1e-8 - 1.0, 1.0 - 1e-8)
        es = np.clip(_sigmoid(es), 1e-8, 1.0 - 1e-8)
        if bias is not None:
            pis = np.where(pis < 0.01, 0.0, pis)
            es = np.where(es < 0.01, 0.0, es)
        return pis, mus, sigmas, rhos, es[:, 0]

    def _sample(self, h3, bias, rng):
        """Draw the next pen sample (dx, dy, pen_up) for every sequence in the batch."""
        pis, mus, sigmas, rhos, es = self._mixture(h3, bias)
        batch = len(h3)
        cum = np.cumsum(pis / pis.sum(axis=1, keepdims=True), axis=1)
        k = np.minimum((cum < rng.random((batch, 1))).sum(axis=1), OUT_K - 1)
        rows = np.arange(batch)
        mu1, mu2 = mus[rows, k], mus[rows, OUT_K + k]
        s1, s2, rho = sigmas[rows, k], sigmas[rows, OUT_K + k], rhos[rows, k]
        z1, z2 = rng.standard_normal(batch), rng.standard_normal(batch)
        dx = mu1 + s1 * z1
        dy = mu2 + s2 * (rho * z1 + np.sqrt(1 - rho * rho) * z2)
        pen = (rng.random(batch) < es).astype(np.float32)
        return np.stack([dx, dy, pen], axis=1).astype(np.float32)

    # ------------------------------------------------------------------ text handling

    @staticmethod
    def _encode(texts):
        codes = [encode_ascii(t) for t in texts]
        lengths = np.array([len(c) for c in codes])
        n = lengths.max()
        onehot = np.zeros((len(codes), n, N_CHARS), dtype=np.float32)
        for i, c in enumerate(codes):
            onehot[i, np.arange(len(c)), c] = 1.0
        mask = (np.arange(n)[None, :] < lengths[:, None]).astype(np.float32)
        return onehot, mask, lengths

    # ------------------------------------------------------------------ writing

    def write(self, texts, biases=None, primes=None, max_steps=None, rng=None, progress=None):
        """Write each text; returns a list of (n, 3) stroke arrays.

        primes: optional list of (strokes, text) style samples, one per text. The
        network first reads the sample while its text is spelled out, which puts
        it in the state of someone who has just written those words in that
        hand, and then carries on with the requested text.
        progress: optional callback(fraction done).
        """
        rng = rng or np.random.default_rng()
        batch = len(texts)
        bias = np.zeros(batch, dtype=np.float32) if biases is None else np.asarray(biases, dtype=np.float32)
        full_texts = [f"{p[1]} {t}" for p, t in zip(primes, texts)] if primes else list(texts)
        onehot, mask, lengths = self._encode(full_texts)
        max_steps = max_steps or 40 * max(len(t) for t in texts)
        state = self.zero_state(batch, onehot.shape[1])

        if primes:
            prime_len = np.array([len(p[0]) for p in primes])
            total = int(prime_len.max()) + max_steps
            for t in range(int(prime_len.max())):
                x = np.stack([p[0][t] if t < len(p[0]) else np.zeros(3, np.float32) for p in primes]).astype(np.float32)
                new = self.step(x, state, onehot, mask)
                live = (t < prime_len)[:, None]
                state = {k: np.where(live, new[k], state[k]) for k in new}
                if progress and t % 25 == 0:
                    progress(t / total)
            x = self._sample(state['h3'], bias, rng)
            done_before = int(prime_len.max())
        else:
            total, done_before = max_steps, 0
            x = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (batch, 1))

        finished = np.zeros(batch, dtype=bool)
        out = [[] for _ in range(batch)]
        for t in range(max_steps):
            # stop a line once attention has reached the end of its text and the pen lifts
            char = state['phi'].argmax(axis=1)
            lift = self._sample(state['h3'], bias, rng)[:, 2] == 1
            finished |= ((char >= lengths - 1) & lift) | (char >= lengths)
            if t > 0:
                x = self._sample(state['h3'], bias, rng)
            if finished.all():
                break
            for i in np.flatnonzero(~finished):
                out[i].append(x[i])
            new = self.step(x, state, onehot, mask)
            live = (~finished)[:, None]
            state = {k: np.where(live, new[k], state[k]) for k in new}
            if progress and t % 25 == 0:
                progress((done_before + t) / total)
        if progress:
            progress(1.0)
        return [np.array(o, dtype=np.float32).reshape(-1, 3) for o in out]

    # ------------------------------------------------------------------ scoring

    def nll(self, strokes, text):
        """Average negative log-likelihood per pen sample of strokes given text (lower = more natural)."""
        onehot, mask, _ = self._encode([text])
        state = self.zero_state(1, onehot.shape[1])
        x, y = strokes[:-1], strokes[1:]
        hs = np.empty((len(x), LSTM), dtype=np.float32)
        for t in range(len(x)):
            state = self.step(x[t][None].astype(np.float32), state, onehot, mask)
            hs[t] = state['h3'][0]
        pis, mus, sigmas, rhos, es = self._mixture(hs)
        mu1, mu2, s1, s2 = mus[:, :OUT_K], mus[:, OUT_K:], sigmas[:, :OUT_K], sigmas[:, OUT_K:]
        d1, d2 = (y[:, 0:1] - mu1) / s1, (y[:, 1:2] - mu2) / s2
        z = d1 ** 2 + d2 ** 2 - 2 * rhos * d1 * d2
        gauss = np.exp(-z / (2 * (1 - rhos ** 2))) / (2 * np.pi * s1 * s2 * np.sqrt(1 - rhos ** 2))
        likelihood = np.maximum((pis * gauss).sum(axis=1), 1e-8)
        pen = np.where(y[:, 2] == 1, es, 1 - es)
        return float(np.mean(-(np.log(likelihood) + np.log(pen))))
