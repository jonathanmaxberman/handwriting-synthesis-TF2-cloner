"""Turn an image of handwriting into a priming style for the model.

The model is primed with *online* handwriting: a time-ordered list of pen
positions, sampled at a tablet's rate, with pen lifts marked. An image has none
of that, so it is reconstructed here:

1. clean the image: flatten transparency, fix polarity (light-on-dark), even
   out lighting, threshold automatically, bring the writing to a standard size;
2. thin the ink to its centerline and turn it into a graph of lines and
   the places where they meet;
3. walk each connected piece of ink the way a pen would: straight on through
   crossings, doubling back along the ink where a hand does (n, m, h, d...);
4. put the strokes in a plausible writing order (words left to right, dots and
   crosses after the word they belong to);
5. resample along the pen path with a human-like speed profile, at the point
   density of the styles the model was trained on (~32 points per character).
"""
import numpy as np
from PIL import Image

from StrokestoNPY import MAX_PRIME_POINTS

INK_HEIGHT_PX = 120        # writing is rescaled to this height before tracing
POINTS_PER_CHAR = 32       # median over the styles shipped with the model
MAX_POINTS = MAX_PRIME_POINTS - 50


# ---------------------------------------------------------------- image cleanup

def _otsu(grey):
    hist = np.bincount(grey.ravel(), minlength=256).astype(float)
    total = hist.sum()
    cum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * np.arange(256))
    with np.errstate(divide='ignore', invalid='ignore'):
        between = (cum_mean[-1] * cum - cum_mean * total) ** 2 / (cum * (total - cum))
    return int(np.nanargmax(between))


def load_ink_mask(image_path, threshold=None):
    """Return a boolean array, True where there is ink, scaled so the writing is INK_HEIGHT_PX tall."""
    import cv2

    image = Image.open(image_path)
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGBA')
        image = Image.alpha_composite(Image.new('RGBA', image.size, (255, 255, 255, 255)), image)
    grey = np.array(image.convert('L'))

    # work at a bounded size so kernel sizes below mean the same thing for every input
    longest = max(grey.shape)
    if longest > 2400:
        scale = 2400 / longest
        grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # light writing on a dark background -> invert
    if np.median(grey) < 128:
        grey = 255 - grey

    if threshold is None:
        # even out shadows / lighting gradients: estimate the paper by closing away
        # the (thin, dark) ink, then express every pixel relative to the paper
        k = max(15, (min(grey.shape) // 4) | 1)
        paper = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        paper = cv2.GaussianBlur(paper, (0, 0), k / 2)
        flat = np.clip(grey.astype(np.float32) / np.maximum(paper.astype(np.float32), 1) * 255, 0, 255).astype(np.uint8)
        flat = cv2.GaussianBlur(flat, (3, 3), 0)
        ink = flat < _otsu(flat)
    else:
        ink = grey < threshold

    # remove specks: connected components far smaller than the largest piece of ink
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    if count <= 1:
        raise ValueError("No writing found in the image.")
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= max(4, 0.01 * areas.max())) + 1
    ink = np.isin(labels, keep)

    # crop to the writing and bring it to a standard height
    ys, xs = np.nonzero(ink)
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = INK_HEIGHT_PX / ink.shape[0]
    size = (max(1, round(ink.shape[1] * scale)), INK_HEIGHT_PX)
    ink = cv2.resize(ink.astype(np.uint8) * 255, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    ink = cv2.GaussianBlur(ink, (0, 0), 1.0) > 127
    return np.pad(ink, 8)


# ---------------------------------------------------------------- skeleton graph

_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _length(line):
    return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())


def skeleton_graph(ink):
    """Thin the ink to one-pixel lines and describe them as a graph.

    Returns (nodes, edges): nodes is a list of (x, y) positions (line ends and
    places where lines meet); edges is a list of dicts with keys 'a', 'b' (node
    indices) and 'pts' (an (n, 2) polyline from node a to node b, y down).
    """
    import cv2
    from skimage.morphology import skeletonize

    skel = skeletonize(ink)
    h, w = skel.shape
    padded = np.pad(skel, 1)
    degree = sum(padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in _NEIGHBOURS) * skel

    # pixels where lines meet come in small clumps; each clump is one node
    count, labels = cv2.connectedComponents((degree >= 3).astype(np.uint8), connectivity=8)
    node_of = {}
    nodes = []
    for label in range(1, count):
        ys, xs = np.nonzero(labels == label)
        nodes.append((xs.mean(), ys.mean()))
        for y, x in zip(ys, xs):
            node_of[(y, x)] = len(nodes) - 1
    for y, x in zip(*np.nonzero((degree <= 1) & skel)):  # line ends and isolated dots
        nodes.append((float(x), float(y)))
        node_of[(y, x)] = len(nodes) - 1

    def neighbours(y, x):
        return [(y + dy, x + dx) for dy, dx in _NEIGHBOURS
                if 0 <= y + dy < h and 0 <= x + dx < w and skel[y + dy, x + dx]]

    edges = []
    visited = set()

    def walk(start, first):
        """Follow a chain of ordinary pixels from a node pixel until the next node."""
        chain = [start, first]
        prev, cur = start, first
        while cur not in node_of:
            visited.add(cur)
            # prefer 4-connected steps so diagonal shortcuts do not skip pixels
            options = [n for n in neighbours(*cur) if n != prev and n not in visited and n not in chain[-3:]]
            options = [n for n in options if n not in node_of or n != start or len(chain) > 2]
            if not options:
                return None
            options.sort(key=lambda n: (n not in node_of, abs(n[0] - cur[0]) + abs(n[1] - cur[1])))
            prev, cur = cur, options[0]
            chain.append(cur)
        return chain

    seen_pairs = set()
    for pixel, node in list(node_of.items()):
        for first in neighbours(*pixel):
            if first in node_of:
                other = node_of[first]
                if other != node and (min(node, other), max(node, other)) not in seen_pairs:
                    # two nodes that touch directly
                    seen_pairs.add((min(node, other), max(node, other)))
                    edges.append({'a': node, 'b': other, 'pts': np.array([nodes[node], nodes[other]], dtype=float)})
                continue
            if first in visited:
                continue
            chain = walk(pixel, first)
            if chain is None:
                continue
            end = node_of[chain[-1]]
            pts = np.array([(x, y) for y, x in chain], dtype=float)
            pts[0], pts[-1] = nodes[node], nodes[end]
            edges.append({'a': node, 'b': end, 'pts': pts})

    # closed loops with no node on them (an isolated "o")
    remaining = skel.copy()
    for y, x in visited:
        remaining[y, x] = False
    for y, x in node_of:
        remaining[y, x] = False
    count, labels = cv2.connectedComponents(remaining.astype(np.uint8), connectivity=8)
    for label in range(1, count):
        ys, xs = np.nonzero(labels == label)
        if len(ys) < 6:
            continue
        start = (ys[np.argmin(xs)], xs[np.argmin(xs)])
        nodes.append((float(start[1]), float(start[0])))
        loop = [start]
        used = {start}
        cur = start
        while True:
            options = [n for n in neighbours(*cur) if n not in used and labels[n] == label]
            if not options:
                break
            options.sort(key=lambda n: abs(n[0] - cur[0]) + abs(n[1] - cur[1]))
            cur = options[0]
            used.add(cur)
            loop.append(cur)
        pts = np.array([(x, y) for y, x in loop] + [(start[1], start[0])], dtype=float)
        edges.append({'a': len(nodes) - 1, 'b': len(nodes) - 1, 'pts': pts})

    return nodes, edges


def prune_spurs(nodes, edges, min_length):
    """Drop short dead-end twigs, which thinning grows at corners and on rough edges."""
    while True:
        deg = np.zeros(len(nodes), dtype=int)
        for e in edges:
            deg[e['a']] += 1
            deg[e['b']] += 1
        keep = []
        dropped = False
        for e in edges:
            dead_end = (deg[e['a']] == 1) != (deg[e['b']] == 1)
            if dead_end and max(deg[e['a']], deg[e['b']]) >= 3 and _length(e['pts']) < min_length and not dropped:
                dropped = True  # one at a time: removing a twig changes its neighbours' degrees
                continue
            keep.append(e)
        edges = keep
        if not dropped:
            return edges


# ---------------------------------------------------------------- pen path reconstruction

def _direction(pts, span=6.0):
    """Unit direction of travel over the first `span` pixels of a polyline."""
    dist = np.linalg.norm(pts - pts[0], axis=1)
    idx = np.flatnonzero(dist >= span)
    v = (pts[idx[0]] if len(idx) else pts[-1]) - pts[0]
    n = np.linalg.norm(v)
    return v / n if n else np.array([1.0, 0.0])


def _start_score(point):
    return point[0] + 0.5 * point[1]   # leftmost, and uppermost among those


def trace_component(nodes, edges, edge_ids, max_retrace):
    """Reconstruct pen strokes for one connected piece of ink.

    The pen starts at the upper-left end, and at every place where lines meet
    carries on along the smoothest unused line. At a dead end it doubles back
    along the ink to the nearest unused line, as a hand does when writing n, m,
    h, d...; if that is further than max_retrace it lifts and starts a new stroke.
    """
    incident = {}
    for i in edge_ids:
        e = edges[i]
        incident.setdefault(e['a'], []).append(i)
        incident.setdefault(e['b'], []).append(i)
    unused = set(edge_ids)

    def leave(edge_id, node):
        e = edges[edge_id]
        return e['pts'] if e['a'] == node else e['pts'][::-1]

    def other_end(edge_id, node):
        e = edges[edge_id]
        return e['b'] if e['a'] == node else e['a']

    def path_to_unused(node):
        """Shortest walk along the ink from node to a node that still has an unused line."""
        import heapq
        best = {node: 0.0}
        heap = [(0.0, node, [])]
        while heap:
            dist, cur, path = heapq.heappop(heap)
            if dist > best.get(cur, np.inf):
                continue
            if cur != node and any(i in unused for i in incident[cur]):
                return dist, path
            for i in incident[cur]:
                nxt = other_end(i, cur)
                nd = dist + _length(edges[i]['pts'])
                if nd < best.get(nxt, np.inf):
                    best[nxt] = nd
                    heapq.heappush(heap, (nd, nxt, path + [(i, cur)]))
        return np.inf, []

    strokes = []
    while unused:
        # start a stroke at a free end if there is one, else anywhere on the unused ink
        candidates = [n for n in incident if sum(i in unused for i in incident[n]) == 1 and len(incident[n]) == 1]
        if not candidates:
            candidates = [n for n in incident if sum(i in unused for i in incident[n]) % 2 == 1]
        if not candidates:
            candidates = [n for n in incident if any(i in unused for i in incident[n])]
        node = min(candidates, key=lambda n: _start_score(nodes[n]))
        points = [np.array([nodes[node]], dtype=float)]
        heading = None
        while True:
            options = [i for i in incident[node] if i in unused]
            if options:
                if heading is None:
                    choice = options[0]
                else:
                    choice = max(options, key=lambda i: float(np.dot(heading, _direction(leave(i, node)))))
                pts = leave(choice, node)
                unused.discard(choice)
                points.append(pts[1:])
                heading = _direction(pts[::-1]) * -1
                node = other_end(choice, node)
                continue
            if not unused:
                break
            dist, path = path_to_unused(node)
            if dist > max_retrace:
                break
            for i, at in path:
                pts = leave(i, at)
                points.append(pts[1:])
                heading = _direction(pts[::-1]) * -1
                node = other_end(i, at)
        strokes.append(np.vstack(points))
    return strokes


def smooth(line, sigma=1.5):
    """Round off the pixel staircase of a skeleton line without moving its ends."""
    from scipy.ndimage import gaussian_filter1d
    if len(line) < 5:
        return line
    out = gaussian_filter1d(line, sigma, axis=0, mode='nearest')
    out[0], out[-1] = line[0], line[-1]
    return out


def trace_pen_strokes(ink, retrace=0.5):
    """Ink mask -> list of pen strokes ((n, 2) polylines, y down), one list entry per pen-down.

    retrace: how far (as a fraction of the writing's height) the pen may double back along
    the ink before it is assumed to have been lifted instead. Low for print, higher for cursive.
    """
    import cv2

    nodes, edges = skeleton_graph(ink)
    stroke_width = 2 * float(np.median(cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)[ink]))
    edges = prune_spurs(nodes, edges, min_length=max(4.0, 1.5 * stroke_width))

    # group edges into connected pieces of ink
    parent = list(range(len(nodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for e in edges:
        parent[find(e['a'])] = find(e['b'])
    groups = {}
    for i, e in enumerate(edges):
        groups.setdefault(find(e['a']), []).append(i)

    strokes = []
    for edge_ids in groups.values():
        strokes.extend(trace_component(nodes, edges, edge_ids, max_retrace=retrace * INK_HEIGHT_PX))

    # isolated dots have no edges at all
    used_nodes = {e['a'] for e in edges} | {e['b'] for e in edges}
    for i, (x, y) in enumerate(nodes):
        if i not in used_nodes:
            strokes.append(np.array([[x, y], [x + 1.0, y + 1.0]]))

    return [smooth(s) for s in strokes]


def order_strokes(strokes, n_words=1):
    """Put strokes in a plausible writing order.

    Words are written left to right. Within a word the main strokes come first,
    left to right, and the small strokes added afterwards (the cross of a t, the
    dot of an i) come last, as most people write them.
    """
    if not strokes:
        return strokes
    oriented = []
    for line in strokes:
        dx, dy = line[-1] - line[0]
        closed = np.hypot(dx, dy) < 3
        if not closed and len(line) < 40:   # short simple stroke: left to right, or top to bottom
            if (dx < 0) if abs(dx) >= 0.5 * abs(dy) else (dy < 0):
                line = line[::-1]
        oriented.append(line)
    oriented.sort(key=lambda l: l[:, 0].min())

    # split into words at the widest horizontal gaps
    right = np.maximum.accumulate([l[:, 0].max() for l in oriented])
    gaps = [oriented[i + 1][:, 0].min() - right[i] for i in range(len(oriented) - 1)]
    breaks = sorted(np.argsort(gaps)[::-1][:max(0, n_words - 1)]) if gaps else []
    breaks = [b for b in breaks if gaps[b] > 0]
    words, start = [], 0
    for b in breaks:
        words.append(oriented[start:b + 1])
        start = b + 1
    words.append(oriented[start:])

    ordered = []
    for word in words:
        longest = max(_length(l) for l in word)
        main, late = [], []
        for line in word:
            others = [o for o in word if o is not line and _length(o) > 2 * _length(line)]
            inside = any(o[:, 0].min() <= line[:, 0].mean() <= o[:, 0].max() for o in others)
            (late if inside and _length(line) < 0.25 * longest else main).append(line)
        ordered.extend(main + late)
    return ordered


def resample(lines, n_points, beta=0.5, k0=5.0, taper=0.5, window=0.04):
    """Resample strokes into about n_points pen samples with a human-like speed profile.

    The model was trained on tablet recordings, where the points bunch up where
    the pen slows down. Evenly spaced points look unnatural to it. Pen speed is
    simulated with the power law of human movement (slower where the path curves:
    speed ~ curvature^-beta) and by slowing down at the start and end of strokes.
    """
    allp = np.vstack(lines)
    height = max(1e-6, np.percentile(allp[:, 1], 97) - np.percentile(allp[:, 1], 3))
    paths, times = [], []
    for line in lines:
        seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
        s = np.r_[0, np.cumsum(seg)]
        if s[-1] <= 0:
            paths.append(line[:1].repeat(2, axis=0))
            times.append(np.array([0.0, 1e-3]))
            continue
        m = max(8, int(s[-1] / (height * 0.01)))
        u = np.linspace(0, s[-1], m)
        p = np.column_stack([np.interp(u, s, line[:, 0]), np.interp(u, s, line[:, 1])]) / height
        ds = (u[1] - u[0]) / height
        w = max(1, int(window / ds))
        d = p[np.minimum(np.arange(m) + w, m - 1)] - p[np.maximum(np.arange(m) - w, 0)]
        angle = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        curvature = np.abs(np.gradient(angle, ds))
        curvature = np.convolve(np.pad(curvature, w, mode='edge'), np.ones(2 * w + 1) / (2 * w + 1), 'valid')
        speed = (curvature + k0) ** (-beta)
        pos = u / height
        speed = speed * (1 - taper * np.exp(-np.minimum(pos, pos[-1] - pos) / 0.08))
        t = np.r_[0, np.cumsum(ds / ((speed[:-1] + speed[1:]) / 2))]
        paths.append(p * height)
        times.append(t)
    dt = sum(t[-1] for t in times) / max(1, n_points)
    out = []
    for p, t in zip(paths, times):
        n = max(2, int(round(t[-1] / dt)) + 1)
        tt = np.linspace(0, t[-1], n)
        out.append(np.column_stack([np.interp(tt, t, p[:, 0]), np.interp(tt, t, p[:, 1])]))
    return out


def strokes_from_ink(ink, text, density=1.0, retrace=0.5):
    """Ink mask -> ordered, resampled pen strokes ((n, 2) arrays, y down)."""
    strokes = trace_pen_strokes(ink, retrace)
    if not strokes:
        raise ValueError("No strokes found in the image. It needs writing that contrasts with its background.")
    strokes = order_strokes(strokes, n_words=len(text.split()))
    n_chars = max(1, len(text.replace(' ', '')))
    return resample(strokes, min(MAX_POINTS, int(POINTS_PER_CHAR * n_chars * density)))


def image_to_strokes(image_path, text, threshold=None, density=1.0, retrace=0.5):
    """Image of one line of handwriting -> ordered pen strokes ((n, 2) arrays, y down)."""
    return strokes_from_ink(load_ink_mask(image_path, threshold), text, density=density, retrace=retrace)


def strokes_to_offsets(strokes):
    """Ordered pen strokes -> the normalized (dx, dy, pen_up) array the model is primed with."""
    from StrokestoNPY import strokes_to_style
    scale = 150.0 / INK_HEIGHT_PX   # strokes_to_style expects a 200px-tall canvas
    return strokes_to_style([{'x': (s[:, 0] * scale).tolist(), 'y': (s[:, 1] * scale).tolist()} for s in strokes])


def tracetovector(image_path=None, input_string=None, path_to_npy=None, name=None, threshold=None, preview=None,
                  hand=None):
    """Trace an image of handwriting and save it as style {path_to_npy}{name}strokes.npy / chars.npy.

    input_string must be the text written in the image. threshold (0-255) overrides
    the automatic ink/paper separation. If hand (a loaded Hand) is given, several
    reconstructions are tried and the one the model handles best is kept.
    Returns (normalized stroke array, legibility score or None).
    """
    from StrokestoNPY import save_style
    score = None
    if hand is not None:
        from style_quality import best_style_from_image
        offsets, score, _ = best_style_from_image(hand, image_path, input_string, threshold)
    else:
        offsets = strokes_to_offsets(image_to_strokes(image_path, input_string, threshold))
    save_style(offsets, input_string, path_to_npy, name, preview=preview)
    return offsets, score
