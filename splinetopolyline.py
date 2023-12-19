import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def approximate_cubic_spline_with_lines(control_points, num_segments):
    """
    Approximates a cubic spline with line segments.

    :param control_points: A list of control points (x, y) defining the cubic spline.
    :param num_segments: Number of line segments to use for approximation.
    :return: A list of line segments approximating the spline.
    """
    # Extract X and Y coordinates from control points
    xs, ys = zip(*control_points)

    # Create a cubic spline interpolation
    cs = CubicSpline(xs, ys)

    # Generate equally spaced X values
    x_values = np.linspace(xs[0], xs[-1], num_segments + 1)

    # Compute Y values
    y_values = cs(x_values)

    # Create line segments
    line_segments = [(x_values[i], y_values[i], x_values[i+1], y_values[i+1]) for i in range(num_segments)]

    return line_segments

# Example usage
control_points = [(0, 0), (1, 2), (2, 3), (3, 2), (4, 0)]  # Example control points
num_segments = 10  # Number of line segments
line_segments = approximate_cubic_spline_with_lines(control_points, num_segments)
print(line_segments)



def plot_line_segments(line_segments):
    """
    Plot the given line segments.

    :param line_segments: A list of line segments, each defined as a tuple (x1, y1, x2, y2).
    """
    for x1, y1, x2, y2 in line_segments:
        plt.plot([x1, x2], [y1, y2], 'b-')  # 'b-' specifies blue line segments

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cubic Spline Approximation with Line Segments')
    plt.grid(True)
    plt.show()

# Assuming you have already computed line_segments using the previous function
plot_line_segments(line_segments)