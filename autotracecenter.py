import numpy as np
from autotrace import Bitmap
from PIL import Image
from scipy.interpolate import CubicSpline

from StrokestoNPY import handle_draw_data


def approximate_cubic_spline_with_dynamic_reduction(control_points, total_points, target_points):
    """Approximate a traced spline with a polyline, thinning it if the whole trace has too many points."""
    if len(control_points) < 2 or total_points <= target_points:
        return control_points  # No reduction possible or not needed

    # Calculate dynamic reduction factor
    reduction_factor = 2 * max(1, int(total_points / target_points))

    # Reduce the number of segments based on the dynamic reduction factor
    num_segments = max(1, len(control_points) // reduction_factor)

    t_values = np.linspace(0, len(control_points) - 1, len(control_points))
    xs, ys = zip(*control_points)

    # Create cubic splines for x and y as functions of 't'
    cs_x = CubicSpline(t_values, xs)
    cs_y = CubicSpline(t_values, ys)

    # Sample points along the spline
    t_samples = np.linspace(0, len(control_points) - 1, num_segments + 1)
    x_samples = cs_x(t_samples)
    y_samples = cs_y(t_samples)

    return list(zip(x_samples, y_samples))


def extract_control_points(spline):
    return [(point.x, point.y) for point in spline.points]


def tracetovector(image_path=None, input_string=None, path_to_npy=None, name=None, threshold=150, preview=None):
    """Trace an image of dark handwriting on a light background into a priming style.

    Uses autotrace's centerline mode to recover pen strokes, approximates the
    splines with polylines, and saves {path_to_npy}{name}strokes.npy / chars.npy.
    input_string must be the text written in the image.
    """
    image = Image.open(image_path)

    # Flatten any transparency onto white, then threshold to a binary bitmap
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGBA')
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    grey_image = image.convert('L')
    bitmap_image = grey_image.point(lambda x: 255 if x > threshold else 0, mode='1')
    image_array = np.array(bitmap_image.convert('RGB'))

    vector = Bitmap(image_array).trace(
        filter_iterations=2,
        error_threshold=1.5,
        despeckle_level=2,
        line_reversion_threshold=0.01,
        line_threshold=1.0,
        corner_surround=4,
        corner_threshold=100.0,
        corner_always_threshold=60.0,
        tangent_surround=5,
        centerline=True,
    )

    total_points = sum(len(extract_control_points(spline)) for path in vector.paths for spline in path.splines)
    target_points = 1199  # the model's priming buffer holds 1200 points

    all_strokes = []
    for path in vector.paths:
        stroke = {'x': [], 'y': []}
        for spline in path.splines:
            control_points = extract_control_points(spline)
            approx_points = approximate_cubic_spline_with_dynamic_reduction(control_points, total_points, target_points)
            for x, y in approx_points:
                stroke['x'].append(int(x))
                stroke['y'].append(int(-y))  # autotrace's y axis points up; canvas y points down
        all_strokes.append(stroke)

    if not all_strokes:
        raise ValueError("No strokes found in the image. It needs dark writing on a light background.")

    return handle_draw_data(all_strokes, input_string, path_to_npy, name, preview=preview)
