from flask import Flask, render_template, request, redirect, send_from_directory, session, flash
import zipfile
import os
from werkzeug.utils import secure_filename
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt
from cell_count import generate_plot  # Import your generate_plot function here

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add a secret key for session

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'zip_file1' not in request.files or 'zip_file2' not in request.files:
        flash('Please select two ZIP files.')
        return redirect(request.url)

    zip_file1 = request.files['zip_file1']
    zip_file2 = request.files['zip_file2']

    if zip_file1.filename == '' or not zip_file1.filename.endswith('.zip'):
        flash(f'Invalid ZIP file: {zip_file1.filename}')
        return redirect(request.url)

    if zip_file2.filename == '' or not zip_file2.filename.endswith('.zip'):
        flash(f'Invalid ZIP file: {zip_file2.filename}')
        return redirect(request.url)

    with TemporaryDirectory() as temp_dir1, TemporaryDirectory() as temp_dir2:
        zip_filename1 = secure_filename(zip_file1.filename)
        zip_filename2 = secure_filename(zip_file2.filename)

        zip_file_path1 = os.path.join(temp_dir1, zip_filename1)
        zip_file_path2 = os.path.join(temp_dir2, zip_filename2)

        zip_file1.save(zip_file_path1)
        zip_file2.save(zip_file_path2)

        # Create folders to extract the contents
        extract_folder1 = os.path.join('extracted_files', zip_filename1.split('.')[0])
        os.makedirs(extract_folder1, exist_ok=True)

        extract_folder2 = os.path.join('extracted_files', zip_filename2.split('.')[0])
        os.makedirs(extract_folder2, exist_ok=True)

        # Extract the contents of the ZIP files
        with zipfile.ZipFile(zip_file_path1, 'r') as zip_ref1, zipfile.ZipFile(zip_file_path2, 'r') as zip_ref2:
            zip_ref1.extractall(extract_folder1)
            zip_ref2.extractall(extract_folder2)

        

        return redirect('/results')

@app.route('/results')
def show_results():
   

        generate_plot('/Users/gmacheta/Documents/GitHub/cellCS/extracted_files/SHSY5Y_Rep_1/SHSY5Y Rep 1','cell_count_1')

        generate_plot('/Users/gmacheta/Documents/GitHub/cellCS/extracted_files/SHSY5Y_Rep_2/SHSY5Y Rep 2','cell_count_2')
        # Create a unique graph filename
       
        
        return render_template('results.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
