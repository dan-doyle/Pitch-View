from flask import Flask, request, jsonify
import os
import uuid
import json
from werkzeug.utils import secure_filename
from homography_calibration.baseline_cameras import extract_pitch_representation

# Create the Flask application
app = Flask(__name__)

@app.route('/get-pitch-representation', methods=['POST'])
def get_pitch_representation():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected for uploading."}), 400

    else:
        if not os.path.exists('./input_images'):
            os.makedirs('./input_images')

        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + '.' + filename.rsplit('.', 1)[1]
        file.save(os.path.join('./input_images', unique_filename))
        
        relative_file_path = os.path.join('./input_images', unique_filename)
        absolute_file_path = os.path.join(os.getcwd(), relative_file_path)
        # Pass the saved image file to the `extract_pitch_representation` function
        pitch_rep = extract_pitch_representation(absolute_file_path)

        # Remove the saved image file after processing
        os.remove(absolute_file_path)
        
        # Return the JSON response
        return jsonify(pitch_rep)


# Run the Flask application if executed directly
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# create an endpoint that takes an image in as a post request, gives it to a function 'extract_pitch_representation' which returns a JSON, then we send this JSON back as part of the response