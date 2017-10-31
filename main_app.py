import os
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import subprocess
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import base64
import json

import sys 
import os
sys.path.append(os.path.abspath("./model"))
from load_with_dataaug import *

app = Flask(__name__)
global model, graph
model, graph = init()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload/", methods=['POST'])
def upload():
    
    
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)        
        return render_template("predict.html", image_name=filename)
    
@app.route("/predict/", methods=['GET','POST'])
def predict():
    
    TempX = request.get_data()
    filename = TempX.decode('utf8').split("/")[-1]
    target = os.path.join(APP_ROOT, 'images/')
    destination = "/".join([target, filename])
    print(destination)
    x = imread(destination, mode='L')
    x = np.invert(x)
    x = imresize(x,(28,28))
    # reshape image data for use in neural network
    x = x.reshape(1,28,28,1)
    with graph.as_default():
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        response = np.array_str(np.argmax(out, axis=1))
        
        return json.dumps(response) #return response
            
            
@app.route("/goback/", methods=['POST'])
def goback():
    return redirect(url_for('index'))

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)    

if __name__ == "__main__":
    #app.run(port=5000, debug=True)
    # This tells your operating system to listen on all public IPs.
    #app.run(host='0.0.0.0')
    app.run(host='192.168.1.9')
