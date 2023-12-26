import numpy as np
from autotrace import Bitmap, VectorFormat
from PIL import Image
from svgpathtools import svg2paths, wsvg, Line, Path, CubicBezier, QuadraticBezier, Arc
import json
from StrokestoNPY import handle_draw_data
from svgpath2mpl import parse_path
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Load an image in PNG format
#image_path = "./Data/a01/a01-000x/a01-000x-00.png"

# Function to approximate a cubic spline with line segments
def approximate_cubic_spline_with_dynamic_reduction(control_points, total_points, target_points):
    if len(control_points) < 2 or total_points <= target_points:
        return control_points  # No reduction possible or not needed

    # Calculate dynamic reduction factor
    reduction_factor = 2*max(1, int(total_points / target_points))

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


# Function to extract control points from a spline
def extract_control_points(spline):
    return [(point.x, point.y) for point in spline.points]

def tracetovector(image_path=None, input_string=None, path_to_npy=None, name=None):
    image = Image.open(image_path)

    # Convert to greyscale so that the image can be thresholded
    grey_image = image.convert('L')

    # Convert to bitmap (binary image: 0 or 255)
    threshold = 150
    bitmap_image = grey_image.point(lambda x: 255 if x > threshold else 0, mode='1')
    rgb_bitmap = bitmap_image.convert('RGB') #
    rgb_bitmap.show()
    # Convert to NumPy array
    image_array = np.array(rgb_bitmap)
    #print(image_array.shape)
    #image_array_int = image_array.astype(int) * 255

    # Use autotrace to trace the bitmap
    bitmap = Bitmap(image_array)
    vector = bitmap.trace(    filter_iterations=2, # This might need adjustment
        error_threshold=1.5, # Set according to your Inkscape settings
        despeckle_level=2,   # Set according to your Inkscape settings
        line_reversion_threshold=0.01, # Default value, adjust as needed
        line_threshold=1.0,  # Default value, adjust as needed
        corner_surround=4,   # Default value, adjust as needed
        corner_threshold=100.0, # Default value, adjust as needed
        corner_always_threshold=60.0, # Default value, adjust as needed
        tangent_surround=5,  # Default value, adjust as needed
        centerline=True,)
    #vector.encode()
    #print(vector.encode('svg', ))
    # Save the vector as an SVG
    vector.save("image.svg")

    # Get an SVG as a byte string (optional)
    #   svg = vector.encode(VectorFormat.SVG)


    # Load the SVG file
    paths, attributes = svg2paths('image.svg')
    #print("paths", len(paths))
    #new_paths = [approximate_curve_with_lines(path) for path in paths]
    #print("new paths", len(new_paths))
    for path in vector.paths:
        print('path:', path)



    #    wsvg(new_paths, attributes=attributes, filename='approximated_svg_file.svg')
    #mpl_path = parse_path(paths)
    #coords = pl_path.to_polygons()

    # Initialize a list to hold all strokes
    #all_strokes = []
    # Iterate through each path


    total_points = sum(len(extract_control_points(spline)) for path in vector.paths for spline in path.splines)
    target_points = 1199  # Adjust as needed, just below 1200



    all_strokes = []

    # Iterate through each path in the vector
    for path in vector.paths:
        stroke = {'x': [], 'y': []}
        for spline in path.splines:  # Assuming each Path object has a 'splines' attribute
            control_points = extract_control_points(spline)
            #num_segments = 5 * (len(control_points) - 1)  # 5 line segments between each control point
            approx_points = approximate_cubic_spline_with_dynamic_reduction(control_points, total_points, target_points)
            for x, y in approx_points:
                stroke['x'].append(int(x))
                stroke['y'].append(int(-y))  # Negating y if needed, as per your original code
        all_strokes.append(stroke)




    # Print the number of elements (strokes) in the JSON
    print("Number of strokes:", len(all_strokes))
    # Convert the all_strokes data to JSON format
    #all_strokes.pop()
    json_data = json.dumps(all_strokes)
    print(json_data)
    #print(type(json_data))

    # send to be processed to correct format
    handle_draw_data(all_strokes,input_string, path_to_npy, name)

def extract_control_points(spline):
    return [(point.x, point.y) for point in spline.points]



#tracetovector('./Data/lines/a01/a01-000u/a01-000u-00.png', "hello", './', name='0101-a')