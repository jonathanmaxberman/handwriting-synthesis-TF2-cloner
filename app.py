from flask import Flask, request, render_template, url_for, jsonify
#import json
import markdown
from bs4 import BeautifulSoup
from handwriting_synthesis.hand.Hand import Hand
import re
import xml.etree.ElementTree as ET
import time
from StrokestoNPY import handle_draw_data
app = Flask(__name__)

def remove_initial_m0(svg_path):
    # Load the SVG file
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Namespace for SVG
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    # Regex to match 'M0,0' or variations
    m0_pattern = re.compile(r'^\s*M\s*0\s*,?\s*0\s+')

    # Iterate through all 'path' elements
    for path in root.findall('.//svg:path', ns):
        d = path.get('d')
        # Remove the initial 'M0,0' if present
        new_d = m0_pattern.sub('', d)
        path.set('d', new_d)

    # Save the modified SVG
    tree.write(svg_path)


def markdown_to_text(md_content):
    lines = md_content.split('\n')
    text = []
    bold_lines = []

    for line in lines:
        # Convert each line from Markdown to HTML
        html_line = markdown.markdown(line)
        soup = BeautifulSoup(html_line, features="html.parser")

        # Extract text and determine if it's bold
        is_bold = bool(soup.find('strong'))
        line_text = soup.get_text(strip=True)

        # Append results to lists
        bold_lines.append(is_bold)
        text.append(line_text)

    return '\n'.join(text), bold_lines





def write_handwriting_from_markdown(md_content, filename, biases, styles, left_justify, stroke_color):
    """Generate handwriting from Markdown content."""
    print(styles)
    text, bold_lines = markdown_to_text(md_content)
    lines = text.split('\n')
    stroke_widths = [2 if is_bold else 1 for is_bold in bold_lines]
    print(text)
    # Ensure biases and styles lists are the same length as lines
    # If not, repeat or truncate the biases and styles to match the number of lines
    biases = (biases * len(lines))[:len(lines)]
    styles = (styles * len(lines))[:len(lines)]
    print(biases,styles)
    stroke_colors = [stroke_color for _ in lines]  # Fixed color as black
    print(filename, lines, biases, styles, stroke_colors, stroke_widths, left_justify)
    #stroke_widths = [1 for _ in lines]        # Fixed width as 1
    #print(left_justify)
    hand = Hand()
    print(filename, lines, biases, styles, stroke_colors, stroke_widths, left_justify)
    hand.write(
        filename=filename,
        lines=lines,
        biases=biases,
        styles=styles,
        stroke_colors=stroke_colors,
        stroke_widths=stroke_widths,
        left_justify=left_justify
    )
    #remove_initial_m0(filename)

@app.route('/capture_strokes', methods=['POST'])
def capture_strokes():
    try:
        data = request.get_json()
        print("Received data:", data)  # Print the received data
        strokes = data['strokes']
        priming_sequence = data['priming_sequence']
        print("strokes:", strokes)
        print("Priming Sequence:", priming_sequence)
        handle_draw_data(strokes, priming_sequence, "./model/style/","style-16-")
        return jsonify({'status': 'success'})


    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    image_file = None
    if request.method == 'POST':
        md_content = request.form['md_text']
        left_justify = request.form.get('justify', 'left')
        stroke_color = request.form.get('stroke_color', 'black')

        biases = [float(request.form.get('bias', 0.95))]
        styles = [int(request.form.get('style', 16))]

        # Repeat the bias and style for each line
        lines = md_content.split('\n')
        biases *= len(lines)
        styles *= len(lines)

        output_file = 'static/handwriting_output.svg'
        print("writing")
        write_handwriting_from_markdown(md_content, output_file, biases, styles, left_justify, stroke_color)
        print ("wrote")
        timestamp = int(time.time())
        image_file = url_for('static', filename='handwriting_output.svg') + f'?v={timestamp}'

    return render_template('index.html', image_file=image_file)


if __name__ == '__main__':
    app.run(debug=True)
