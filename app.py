from flask import Flask, render_template, request, redirect, send_from_directory, session
import zipfile
import os
from werkzeug.utils import secure_filename
from multiprocessing import Process






app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'zip_file' not in request.files:
        return redirect(request.url)

    file = request.files['zip_file']

    if file.filename == '':
        return redirect(request.url)

    if file and file.filename.endswith('.zip'):
        zip_filename = secure_filename(file.filename)
        session['uploaded_zip_filename'] = zip_filename

        # Save the uploaded ZIP file
        file.save(zip_filename)

        # Create a folder to extract the contents
        extract_folder = 'extracted_files'
        os.makedirs(extract_folder, exist_ok=True)

        # Extract the contents of the ZIP file
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

        return redirect('/results')

@app.route('/results')
def show_results():
    import cv2
    import numpy as np
    import glob
    from cell_count import generate_plot
    
    import matplotlib.pyplot as plt
   
    uploaded_zip_filename = session.get('uploaded_zip_filename', None)
    folder_name = os.path.splitext(uploaded_zip_filename)[0]
    folder_path = os.path.join('extracted_files/', folder_name)
    
    
    generate_plot ()
    
    

  
    return render_template('results.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.secret_key = 'your_secret_key'  # Add a secret key for session
    app.run(debug=True)
