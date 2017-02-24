# -*- coding: utf-8 -*-
"""
Plant Classification webpage auxiliary functions

Author: Ignacio Heredia
Date: December 2016
"""
import numpy as np
import os


def results_html_display(pred_lab, pred_prob):
    """
    Returns html string with the display of the predictions and the prediction probability.

    Parameters
    ----------
    pred_lab : list of strings
        Prediction labels
    pred_prob : list of loats
        Prediction probabilities

    """
    display = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <title>Plant app</title>
    <link type= "text/css" rel="stylesheet" href="static/css/style.css" }}">
    </head>
    <body>
        <div class="title">
            <h2>Top predicted labels</h2>
        </div>
        <div class="container">"""
    for n, p in enumerate(pred_lab):
        display += '              <p>{}. {} | {:.0f} %<p>'.format(n+1, p, pred_prob[n]*100)
    display += """
        </div>
        <br><br><br>
        <form action="/">
            <div class="new-query">
            <button name="newquery-button" type="submit">New query</button>
            </div>
        </form>
        </body>
        </html>
    """
    return display.replace('/n', '')  # replace linebreaks


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
    <meta charset="utf-8">
    <title>Plant app</title>
    <link type= "text/css" rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    </head>
    <body>
    <div class="title">
        <h3>Label list</h3>
    </div>
    <div class="container">"""
    homedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    labels = np.genfromtxt(os.path.join(homedir, 'model_files', 'data', labels_file), dtype='str', delimiter='/n')
    labels = np.insert(labels, np.arange(len(labels)) + 1, '<br>')
    display += " ".join(labels)
    display += """
    </div>
    </body>
    </html>
    """
    with open("templates/label_list.html", "w") as text_file:
        text_file.write(display)
