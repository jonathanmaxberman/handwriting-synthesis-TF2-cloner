import os

# Flask handles requests off the main thread; matplotlib must not try to open windows there.
os.environ.setdefault("MPLBACKEND", "Agg")

import re
import tempfile
import threading
import time

import markdown
from bs4 import BeautifulSoup
from flask import Flask, request, render_template, url_for, jsonify

from handwriting_synthesis import Hand
from handwriting_synthesis.config import style_path
from StrokestoNPY import handle_draw_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)  # model/ and static/ paths are relative to the repo root

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

STYLE_DIR = os.path.join(style_path, "")  # trailing separator
DEFAULT_CUSTOM_STYLE = 16

_hand = None
_hand_lock = threading.Lock()


def get_hand():
    """Load the model once and reuse it; building the graph takes several seconds."""
    global _hand
    with _hand_lock:
        if _hand is None:
            _hand = Hand()
        return _hand


def available_styles():
    styles = []
    for filename in os.listdir(style_path):
        match = re.fullmatch(r'style-(\d+)-strokes\.npy', filename)
        if match and os.path.exists(os.path.join(style_path, f'style-{match.group(1)}-chars.npy')):
            styles.append(int(match.group(1)))
    return sorted(styles)


def markdown_to_text(md_content):
    """Strip Markdown from each line; lines containing **bold** are flagged."""
    text = []
    bold_lines = []
    for line in md_content.split('\n'):
        soup = BeautifulSoup(markdown.markdown(line), features="html.parser")
        bold_lines.append(bool(soup.find('strong')))
        text.append(soup.get_text(strip=True))
    return text, bold_lines


def write_handwriting_from_markdown(md_content, filename, bias, style, justify, stroke_color):
    """Generate handwriting from Markdown content."""
    lines, bold_lines = markdown_to_text(md_content)
    with _hand_lock_for_sampling:
        get_hand().write(
            filename=filename,
            lines=lines,
            biases=[bias] * len(lines),
            styles=[style] * len(lines),
            stroke_colors=[stroke_color] * len(lines),
            stroke_widths=[2 if is_bold else 1 for is_bold in bold_lines],
            left_justify=justify,
        )


_hand_lock_for_sampling = threading.Lock()


def parse_style_number(value):
    style = int(value)
    if not 0 <= style <= 999:
        raise ValueError("Style number must be between 0 and 999.")
    return style


@app.route('/capture_strokes', methods=['POST'])
def capture_strokes():
    """Save strokes drawn in the browser as a priming style."""
    try:
        data = request.get_json()
        priming_sequence = (data.get('priming_sequence') or '').strip()
        if not priming_sequence:
            raise ValueError("Type the words you wrote into the priming sequence box.")
        style = parse_style_number(data.get('style', DEFAULT_CUSTOM_STYLE))
        preview = f'static/style-{style}-preview.png'
        strokes = handle_draw_data(data['strokes'], priming_sequence, STYLE_DIR, f"style-{style}-", preview=preview)
        return jsonify({'status': 'success', 'style': style, 'points': len(strokes),
                        'preview': url_for('static', filename=f'style-{style}-preview.png') + f'?v={int(time.time())}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Trace an uploaded image of handwriting into a priming style."""
    try:
        from autotracecenter import tracetovector

        upload = request.files.get('image')
        if upload is None or not upload.filename:
            raise ValueError("Choose an image file.")
        priming_sequence = (request.form.get('priming_sequence') or '').strip()
        if not priming_sequence:
            raise ValueError("Type the words written in the image into the priming sequence box.")
        style = parse_style_number(request.form.get('style', DEFAULT_CUSTOM_STYLE))
        threshold = int(request.form.get('threshold', 150))

        suffix = os.path.splitext(upload.filename)[1][:10]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            upload.save(tmp)
        try:
            preview = f'static/style-{style}-preview.png'
            strokes = tracetovector(tmp.name, priming_sequence, path_to_npy=STYLE_DIR, name=f"style-{style}-",
                                    threshold=threshold, preview=preview)
        finally:
            os.unlink(tmp.name)
        return jsonify({'status': 'success', 'style': style, 'points': len(strokes),
                        'preview': url_for('static', filename=f'style-{style}-preview.png') + f'?v={int(time.time())}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/', methods=['GET', 'POST'])
def index():
    image_file = None
    error = None
    form = {'md_text': '', 'bias': '0.95', 'style': str(DEFAULT_CUSTOM_STYLE), 'justify': 'left',
            'stroke_color': 'black'}
    if request.method == 'POST':
        form.update({k: request.form.get(k, v) for k, v in form.items()})
        try:
            md_content = form['md_text'].replace('\r\n', '\n').strip('\n')
            if not md_content.strip():
                raise ValueError("Type some text to write.")
            style = int(form['style'])
            if style not in available_styles():
                raise ValueError(f"Style {style} does not exist yet. Available styles: {available_styles()}")
            os.makedirs('static', exist_ok=True)
            write_handwriting_from_markdown(md_content, 'static/handwriting_output.svg', float(form['bias']), style,
                                            form['justify'], form['stroke_color'])
            image_file = url_for('static', filename='handwriting_output.svg') + f'?v={int(time.time())}'
        except ValueError as e:
            error = str(e)

    return render_template('index.html', image_file=image_file, error=error, form=form, styles=available_styles(),
                           custom_style=DEFAULT_CUSTOM_STYLE)


if __name__ == '__main__':
    # Port 5000 is taken by AirPlay Receiver on macOS. The reloader is off because it
    # would import TensorFlow twice.
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 5001)), debug=True, use_reloader=False)
