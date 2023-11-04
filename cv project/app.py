from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_edges', methods=['POST'])
def detect_edges():
    if 'image' not in request.files:
        return "No file part"

    image = request.files['image']

    if image.filename == '':
        return "No selected file"

    if image:
        image_stream = image.read()
        nparr = np.fromstring(image_stream, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # Save the edges image as bytes
        retval, buffer = cv2.imencode(".jpg", edges)
        edge_image_bytes = buffer.tobytes()

        return send_file(
            io.BytesIO(edge_image_bytes),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='canny_edges.jpg'
        )

if __name__ == '__main__':
    app.run(debug=True)
