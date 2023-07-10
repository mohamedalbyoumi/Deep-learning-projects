from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from main import getPrediction

# Save videos to the 'static' folder as Flask serves files from this directory
UPLOAD_FOLDER = 'static/videos/'
MASK_FOLDER = 'static/masked_videos/'
THRESHOLD_FOLDER = 'static/thresholded_videos/'

# Create an app object using the Flask class
app = Flask(__name__, static_folder="static")

# Add a secret key for cookie encryption
app.secret_key = "1056906343"

# Set the upload and output folders
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MASK_FOLDER'] = MASK_FOLDER
app.config['THRESHOLD_FOLDER'] = THRESHOLD_FOLDER

# Define the video operations
VIDEO_OPERATIONS = {
    'mask': {'apply_mask': True, 'output_folder': 'MASK_FOLDER'},
    'threshold': {'apply_mask': False, 'output_folder': 'THRESHOLD_FOLDER'},
}


@app.route('/')
def index():
    return render_template('index.html', input_video=None, operated_video=None)


# Define the route for file submission
@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            operation = request.form.get('operation')
            if operation not in VIDEO_OPERATIONS:
                flash('Invalid operation')
                return redirect(request.url)

            operation_params = VIDEO_OPERATIONS[operation]
            apply_mask = operation_params['apply_mask']
            output_folder = operation_params['output_folder']

            output_path = os.path.join(app.config[output_folder], operation + '_' + filename)
            # Call the getPrediction function passing the video path, output path, and apply_mask parameter
            getPrediction(video_path, output_path, apply_mask)
            flash('Masked video created successfully.')

            # Pass the video paths to the template
            input_video = '/' + video_path
            operated_video = '/' + output_path
            return render_template('index.html', input_video=input_video, operated_video=operated_video)


if __name__ == "__main__":
    app.run()
