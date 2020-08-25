import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/uploads'
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/downloads'
ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__, static_url_path="/static")
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

# limit upload size to 8MB
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

OUTPUT_FILE = DOWNLOAD_FOLDER + "/output.txt"
@app.route("/downloads/output.txt", methods=['GET', 'POST'])

def output():
    #print(OUTPUT_FILE)
    f = open(OUTPUT_FILE, "r")
    return f.read()
    #return redirect("/downloads/output.txt")

@app.route('/success.html', methods=['GET', 'POST'])

def main():
    if request.method == 'POST':
        return redirect("/")
    return render_template('success.html')

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("No file attached in request")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print("No file selected")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            process_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename)
            return redirect("success.html")
            #return redirect(url_for('uploaded_file', filename=filename))
    return render_template('index.html')

def process_file(path, filename):
    perform_ocr(path, filename)

import ocr
def perform_ocr(path, filename):
    ocr.ocr(path)

if __name__ == '__main__':
    #port = int(os.environ.get("PORT", 8112))
    app.run(debug=True, host='0.0.0.0')