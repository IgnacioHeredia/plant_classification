# -*- coding: utf-8 -*-
"""
Plant Classification webpage auxiliary functions

Author: Ignacio Heredia
Date: December 2016
"""
import numpy as np
import os, sys
import requests
import json
from flask import flash, Markup
from werkzeug import secure_filename
import random

homedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(homedir)
from model_files.test_utils import load_model, single_prediction


# Loading label names and label info files 
synsets = np.genfromtxt(os.path.join(homedir, 'model_files', 'data', 'synsets.txt'), dtype='str', delimiter='/n')
try:
    synsets_info = np.genfromtxt(os.path.join(homedir, 'model_files', 'data', 'info.txt'), dtype='str', delimiter='/n')
except:
    synsets_info = np.array(['']*len(synsets))
assert synsets.shape == synsets_info.shape, """
Your info file should have the same size as the synsets file.
Blank spaces corresponding to labels with no info should be filled with some string (eg '-').
You can also choose to remove the info file."""
    
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


def catch_url_error(url_list):
    error_dict = {}
    
    # Error catch: Empty query
    if not url_list:
        error_dict['Error_type'] = 'Empty query'
        return error_dict
           
    for i in url_list:    
        
        # Error catch: Inexistent url        
        try:
            url_type = requests.head(i).headers.get('content-type')
        except:
            error_dict['Error_type'] = 'Failed url connection'
            error_dict['Error_description'] = """Check you wrote the url address correctly."""
            return error_dict
            
        # Error catch: Wrong formatted urls    
        if url_type.split('/')[0] != 'image':
            error_dict['Error_type'] = 'Url image format error'
            error_dict['Error_description'] = """Some urls were not in image format. Check you didn't uploaded a preview of the image rather than the image itself."""
            return error_dict
        
    return error_dict


def allowed_file(app, filename):
    """
    For a given file, return whether it's an allowed type or not
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def catch_localfile_error(app, local_list):
    error_dict = {}
    
    # Error catch: Empty query
    if not local_list[0].filename:
        error_dict['Error_type'] = 'Empty query'
        return error_dict
    
    # Error catch: Image format error
    for f in local_list:
        if not allowed_file(app, f.filename):
            error_dict['Error_type'] = 'Local image format error'
            error_dict['Error_description'] = """At least one file is not in a standard image format (jpg|jpeg|png)."""
            return error_dict
        
    return error_dict


def print_error(app, message):
    app.logger.error(message['Error_type'])
    error_message = '<center><b>{}</b></center>'.format(message['Error_type'])
    if 'Error_description' in message.keys():
        error_message += '<br>{}'.format(message['Error_description'])
    flash(Markup(error_message))
    
    
def image_link(pred_lab):
    """
    Return link to Google images
    """
    base_url = 'https://www.google.es/search?'
    links = []
    for i in pred_lab:
        params = {'tbm':'isch','q':i}
        url = base_url + requests.compat.urlencode(params) 
        links.append(url)
    return links


def wikipedia_link(pred_lab):
    """
    Return link to wikipedia webpage
    """
    base_url = 'https://en.wikipedia.org/wiki/'
    links = []
    for i in pred_lab:
        url = base_url + i.replace(' ', '_')
        links.append(url)
    return links
    

def successful_message(pred_lab, pred_prob):
    
    lab_name = synsets[pred_lab].tolist()
    
    message = {'pred_lab': lab_name,   
               'pred_prob':pred_prob.tolist(),
               'google_images_link': image_link(lab_name),
               'wikipedia_link': wikipedia_link(lab_name),
               'info': synsets_info[pred_lab].tolist(),
               'status': 'OK'}
    return message


def url_prediction(url_list):
    
    # Catch errors (if any)
    error_message = catch_url_error(url_list)
    if error_message:
        message = {'status': 'error'}
        message.update(error_message)      
        return message
    
    # Predict
    pred_lab, pred_prob = single_prediction(test_func, im_list=url_list, aug_params={'mean_RGB': mean_RGB, 'filemode':'url'})

    return successful_message(pred_lab, pred_prob)


def localfile_prediction(app, uploaded_files):
    

    # Catch errors (if any)
    error_message = catch_localfile_error(app, uploaded_files)
    if error_message:
        message = {'status': 'error'}
        message.update(error_message)      
        return message
    
    # Save images
    filenames = []
    for f in uploaded_files:
#        filename = secure_filename(f.filename)
        filename = str(random.randint(0, 1000000000))
        file_path = os.path.join(homedir, 'webpage_files', 'templates', 'uploads', filename)
        f.save(file_path)
        filenames.append(file_path)
                        
    # Predict
    pred_lab, pred_prob = single_prediction(test_func, im_list=filenames, aug_params={'mean_RGB': mean_RGB, 'filemode':'local'})

    # Remove cache images
    for f in filenames:
        os.remove(f)

    return successful_message(pred_lab, pred_prob)


def label_list_to_html(labels_file):
    """
    Transform the labels_list.txt to an html file to show as database.

    Parameters
    ----------
    labels_file : path to txt file
        Name of labels file (synsets.txt)

    """
    display = """
    <!DOCTYPE html>
    <html lang="en">
    
    <head>
        <!-- Basic Page Needs
        –––––––––––––––––––––––––––––––––––––––––––––––––– -->
        <meta charset="utf-8">
        <title>Plant App</title>
        <meta name="description" content="">
        <meta name="author" content="">
        
        <!-- Mobile Specific Metas
        –––––––––––––––––––––––––––––––––––––––––––––––––– -->
        <meta name="viewport" content="width=device-width, initial-scale=1">
        
        <!-- FONT
        –––––––––––––––––––––––––––––––––––––––––––––––––– -->
        <link href="//fonts.googleapis.com/css?family=Raleway:400,300,600" rel="stylesheet" type="text/css">
        
        <!-- CSS
        –––––––––––––––––––––––––––––––––––––––––––––––––– -->
        <link type= "text/css" rel="stylesheet" href="../static/css/normalize.css">
        <link type= "text/css" rel="stylesheet" href="../static/css/skeleton.css">
        <link type= "text/css" rel="stylesheet" href="../static/css/general.css">
        <link type= "text/css" rel="stylesheet" href="../static/css/custom.css">
            
    </head>
    
    <body>
    
    <!-- Primary Page Layout
    –––––––––––––––––––––––––––––––––––––––––––––––––– -->
    <div class="container">
        <section class="header">
            <h1 class="center"  style="margin-top: 25%">Automated Plant Recognition</h1>
        </section>
        <div class="docs-section">
            <div class="row">
              <div class="eight columns offset-by-two column">
                  
                  <h4>Labels</h4>
                  <p id="show_predictions"></p>
                  <script>document.getElementById("show_predictions").innerHTML = text;</script>
                  <br>"""
    homedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    labels = np.genfromtxt(os.path.join(homedir, 'model_files', 'data', labels_file), dtype='str', delimiter='/n')
    labels = np.insert(labels, np.arange(len(labels)) + 1, '<br>')
    display += " ".join(labels)                
    display += """ 
              </div>  
          </div>
        </div>
    </div>
    </body>
    </html>"""
  
    with open("templates/label_list.html", "w") as text_file:
        text_file.write(display)
