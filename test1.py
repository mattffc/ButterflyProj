import os
from flask import Flask, request, redirect, url_for
from werkzeug import secure_filename
from leNetpack import logistic_sgd
from leNetpack import mlp
from leNetpack import LeNet5WEB

UPLOAD_FOLDER = r'C:\Users\Matt\Desktop\WebDev\uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
#WSGIApplicationGroup %{GLOBAL}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
from flask import send_from_directory
from flask import request
@app.route('/test')
def tester():
    print('blahggg')
    a,b,c = LeNet5WEB.predict(1)
    print('blargh')
    print(a)
    return str(a)
def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    
@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    img = os.path.join(app.config['UPLOAD_FOLDER'],filename)
    print(img)
    a,b,c = LeNet5WEB.predict(1,img=img)
    print(app.config['UPLOAD_FOLDER'],filename)
    return str(a)#send_from_directory(app.config['UPLOAD_FOLDER'],filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    print('hh')
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''  
    
    
if __name__ == '__main__':
    app.run()