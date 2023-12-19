import xml.etree.ElementTree as ET
import numpy as np

def svg_path_to_lines(svg_file_path, precision=0.1):
    """
    Convert SVG path data to a series of line segments.

    Args:
    svg_file_path (str): Path to the SVG file.
    precision (float): The precision used for approximating curves with line segments.

    Returns:
    list of tuples: List of (x, y) coordinates representing line segments.
    """
    lines = []

    # Parse the SVG file
    tree = ET.parse(svg_file_path)
    root = tree.getroot()

    # Define functions to parse SVG path commands
    def parse_coordinates(coord_string):
        coords = coord_string.replace(',', ' ').split()
        return float(coords[0]), float(coords[1])

    def parse_path(path_string):
        commands = path_string.split()
        command = commands[0]
        coords = [parse_coordinates(coord) for coord in commands[1:]]
        return command, coords

    # Iterate through path elements in the SVG
    for path_element in root.findall(".//{http://www.w3.org/2000/svg}path"):
        path_data = path_element.get("d")

        current_point = (0, 0)
        for command, coords in [parse_path(path_data)]:
            for coord in coords:
                if command == 'M':
                    current_point = coord
                elif command == 'L':
                    lines.append((current_point, coord))
                    current_point = coord
                elif command == 'C':
                    # Approximate curves with line segments
                    p0, p1, p2 = current_point, coords[0], coords[1]
                    t = 0
                    while t < 1:
                        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
                        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
                        lines.append((current_point, (x, y)))
                        current_point = (x, y)
                        t += precision
                command = 'L'  # Convert curves to lines for simplicity

    return lines

# Example usage:
lines = svg_path_to_lines("image.svg", precision=0.05)
print(lines)
