# -*- coding: utf-8 -*-
"""
Plant Classification webpage

Author: Ignacio Heredia
Date: December 2016

Descrition:
This script launches a basic webpage interface to return results on the plant classification.
To launch the webpage, enter in Ubuntu terminal:
    export FLASK_APP=plant_webpage_demo.py
    python -m flask run

"""

from flask import Flask, render_template, request, send_from_directory
from werkzeug import secure_filename
import os
import sys
import json
import numpy as np
from webpage_utils import results_html_display, label_list_to_html
homedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(homedir)
from model_files.test_utils import load_model, single_prediction

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

metadata = np.genfromtxt(os.path.join(homedir, 'model_files', 'data', 'synsets.txt'), dtype='str', delimiter='/n')

# Load training info
info_files = os.listdir(os.path.join(homedir, 'model_files', 'training_info'))
info_file_name = [i for i in info_files if i.endswith('.json')][0]
info_file = os.path.join(homedir, 'model_files', 'training_info', info_file_name)
with open(info_file) as datafile:
    train_info = json.load(datafile)
mean_RGB = train_info['augmentation_params']['mean_RGB']
output_dim = train_info['training_params']['output_dim']

# Load net weights 
weights_files = os.listdir(os.path.join(homedir, 'model_files', 'training_weights'))
weights_file_name = [i for i in weights_files if i.endswith('.npz')][0]
test_func = load_model(os.path.join(homedir, 'model_files', 'training_weights', weights_file_name), output_dim=output_dim)

# Create labels.html from synsets.txt
label_list_to_html(os.path.join(homedir, 'model_files', 'data', 'synsets.txt'))


@app.route('/')
def my_form():
    return render_template("main_page.html")


@app.route('/label_list.html/')
def label_list():
    return render_template('label_list.html')


@app.route('/robots.txt')
def static_from_root():
    return send_from_directory(app.static_folder, request.path[1:])


@app.route('/url_upload', methods=['POST'])
def my_form_post():
    url = request.form['url']
    url = [i.replace(' ', '') for i in url.split(' ') if i != '']
    pred_lab, pred_prob = single_prediction(test_func, im_list=url, aug_params={'mean_RGB': mean_RGB, 'filemode':'url'})
    return results_html_display(metadata[pred_lab], pred_prob)


def allowed_file(filename):
    '''
    For a given file, return whether it's an allowed type or not
    '''
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/local_upload', methods=['POST'])
def local_post():
    uploaded_files = request.files.getlist("local_files")
    filenames = []
    for f in uploaded_files:
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            file_path = os.path.join(homedir, 'webpage_files', 'templates', 'uploads', filename)
            f.save(file_path)
            filenames.append(file_path)
    pred_lab, pred_prob = single_prediction(test_func, im_list=filenames, aug_params={'mean_RGB': mean_RGB, 'filemode':'local'})
    for f in filenames:
        os.remove(f)
    return results_html_display(metadata[pred_lab], pred_prob)

if __name__ == '__main__':
    app.debug = False
    app.run()
