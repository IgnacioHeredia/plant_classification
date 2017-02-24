# -*- coding: utf-8 -*-
"""
PlantNet_test_url_via_upload

Author: Ignacio Heredia
Date: November 2016

Description:
Script to test the identification performance of the PlantNet tool when
identiying  some image url.
This script can be runned in two ways:
    * use_via_upload = False: we send a query to identify directly the url.
    * use_via_upload = True: we download the image from the url to our
    computer and then upload the image to PlantNet for querying. This method
    is more stable (less query errors) than directly predicting the url

If you want to use proxies (use_proxies=True) make sure to run before the
proxy generators in the scrapping pool folder to have freshly harvested proxies.

This script runs in a parallel fashion to send queries using as many
threads in the computer as possible using threading.Treads().
"""

import urllib
import requests
import json
import numpy as np
import os
import time
import threading
from StringIO import StringIO


homedir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Select dataset
dataset = 'google'
print("Loading url data...")
metadata = np.genfromtxt(os.path.join(homedir, 'data', 'data_splits', 'synsets.txt'), dtype='str', delimiter='/n')
metadata_binomial = np.genfromtxt(os.path.join(homedir, 'data', 'data_splits', 'synsets_binomial.txt'), dtype='str', delimiter='/n')
if dataset == 'ptflora':
    outfile_name = 'plantnettool_predictions_ptflora.json'
    im_list = np.genfromtxt(os.path.join('test_datasets', 'test_ptflora.txt'), dtype='str', delimiter=' ; ')
    X, y = im_list[:, 0], np.array([list(metadata_binomial).index(label) for label in im_list[:, 1]]).astype(np.int64)
if dataset == 'inaturalist':
    outfile_name = 'plantnettool_predictions_inaturalist.json'
    im_list = np.genfromtxt(os.path.join('test_datasets', 'test_inaturalist.txt'), dtype='str', delimiter=' ; ')
    X, y = im_list[:, 0], np.array([list(metadata_binomial).index(label) for label in im_list[:, 1]]).astype(np.int64)
    X = np.array([i.replace('original', 'medium') for i in X])
if dataset == 'google':
    outfile_name = 'plantnetool_predictions_google.json'
    im_list = np.genfromtxt(os.path.join('test_datasets', 'test_google_image_search.txt'), dtype='str', delimiter=' ; ')
    X, y = im_list[:, 0], np.array([list(metadata_binomial).index(label) for label in im_list[:, 1]]).astype(np.int64)

np.random.seed(101)
subset_size = 10000
ind = np.random.choice(range(len(y)), size=subset_size, replace=False)
X, y = X[ind], y[ind]
pred_dict = {'pred_lab': [], 'pred_prob': [], 'true_lab': []}
tags = ['flower', 'fruit', 'leaf', 'bark']  # tag 'habit' almost always gives error
base_url = 'http://identify.plantnet-project.org/api/project/weurope/identify?'

max_threads = 20  # parallel threads
use_via_upload = True
use_proxies = True
if use_proxies:
    proxy_pool = np.genfromtxt(os.path.join('scrapping_pools', 'proxylist2.txt'), dtype='str', delimiter='/n')
    useragent_pool = np.genfromtxt(os.path.join('scrapping_pools', 'useragents_list.txt'), dtype='str', delimiter='/n')
    s = requests.Session()
    a = requests.adapters.HTTPAdapter(pool_connections=len(proxy_pool), pool_maxsize=len(proxy_pool), max_retries=1)
    s.mount('http://', a)


def download_and_upload(img_url):
    """
    Donwloads image from url and uploads the image again. It returns the new url
    of the uploaded image.
    """
    im = requests.get(img_url)  # download image
    post = requests.post('http://identify.plantnet-project.org/api/mupload', files={'im1': StringIO(im.content)})  # post image from our local disk
    new_photo_link = post.json()['map']['im1']  # get link of posted image
    new_img_url = 'http://bs.plantnet-project.org/img/' + new_photo_link
    return new_img_url


def prediction(image_list, image_labels, acc_dict):

    for n, (img_url, label) in enumerate(zip(image_list, image_labels)):
        print 'Image {}'.format(n)
        tmp = {'pred_lab': [], 'pred_prob': []}
        if use_via_upload:
            try:
                img_url = download_and_upload(img_url)
            except:
                print 'Upload error'
                continue
        for tag in tags:
            params = {'imgs': img_url, 'tags': tag, 'json': 'True', 'lang': 'en', 'app_version': 'web-1.0.0'}  # query parameters
            url = base_url + urllib.urlencode(params)
            try:
                # time.sleep(np.random.rand()*5)
                if use_proxies:
                    r = s.get(url, proxies={'http': np.random.choice(proxy_pool)}, headers={'User-Agent': np.random.choice(useragent_pool), 'Connection': 'close'})
                else:
                    r = requests.get(url)
                if r.status_code == 200:
                    query = r.json()
                    results = query['results']
                    pred_lab = [results[i]['sp'] for i in range(5)]  # top5 predicted labels
                    for i, p in enumerate(pred_lab):
                        try:
                            pred_lab[i] = list(metadata).index(p.encode('ascii', 'ignore'))
                        except:  # label not present in our training set
                            pred_lab[i] = np.random.randint(0, len(metadata))
                    pred_prob = [float(results[i]['score'])/100. for i in range(5)]  # top5 predicted labels' probabilities
                    tmp['pred_lab'].append(pred_lab)
                    tmp['pred_prob'].append(pred_prob)
                else:
                    print 'Error with tag {} - Error status code {}'.format(tag, r.status_code)
            except Exception as e:
                print 'Error with tag {} - Error info: {}'.format(tag, e)

        # We select best accuracy results across tags
        # (ie. we suppose the user selects optimally the tag).
        if tmp['pred_lab']:  # at least one tag has returned a result
            clas = np.argwhere(np.array(tmp['pred_lab']) == label)
            if clas.size:  # at least one tag has the correct label in its top5 predictions
                args = np.argsort(clas[:, 1])
                i = args[0]
            else:  # select randomly one tag's prediction
                i = np.random.randint(0, len(tmp['pred_lab']))
            pred_dict['pred_lab'].append(tmp['pred_lab'][i])
            pred_dict['pred_prob'].append(tmp['pred_prob'][i])
            pred_dict['true_lab'].append(label)


def run_parallel(image_list, image_labels, pred_dict, thread_num=5, tmax=500.):

    t_list = []
    for args in np.array_split(np.arange(len(image_list)), thread_num):
        X_tmp, y_tmp = image_list[args], image_labels[args]
        t = threading.Thread(target=prediction, args=(X_tmp, y_tmp, pred_dict,))
        t_list.append(t)

    for t in t_list:
        t.daemon = True
        time.sleep(1.)
        t.start()

    t0 = time.time()
    for t in t_list:
        dt = time.time() - t0
        t.join(max(tmax-dt, 0.))


def main_parallel(image_list, image_labels, pred_dict, thread_num=5,
                  batchsize=100, tmax=1000.):
    """
    We process in several steps to save intermediate results in case of failure.

    Parameters
    ----------
    image_list : array of strs
        List of urls to query
    image_labels : array of ints
        Label number of the images
    pred_dict : dict
        Storage of the prediction label and probabilities.
    thread_num : int
        Number of threads to use
    tmax : float
        Maximal time allowed to the program to finish the task. We set this so
        that the whole script isn't blocked if a thread gets stucked for some reason.
        Should be ~ batchsize / (thread_num * internet_speed). As estimation,
        processing 50 images via public proxies with 20 threads takes close to 10 min.
    batchsize : int
        Number of images to process before doing a backup

    """
    for args in np.array_split(np.arange(len(image_list)), int(len(image_list)*1./batchsize)):
        X_tmp, y_tmp = image_list[args], image_labels[args]
        run_parallel(X_tmp, y_tmp, pred_dict, thread_num, tmax)
        with open(os.path.join('test_predictions', 'temp_' + outfile_name), 'wb') as f:
            json.dump(pred_dict, f)

main_parallel(X, y, pred_dict, thread_num=max_threads)

# Saving predictions
with open(os.path.join('test_predictions', outfile_name), 'wb') as f:
        json.dump(pred_dict, f)

# Print the accuracy
top1, top5 = 0, 0
for i, p in enumerate(pred_dict['true_lab']):
    if p == pred_dict['pred_lab'][i][0]:
        top1 += 1
    if p in pred_dict['pred_lab'][i]:
        top5 += 1
print 'Top1 Accuracy: {}'.format(1. * top1 / len(pred_dict['true_lab']))
print 'Top5 Accuracy: {}'.format(1. * top5 / len(pred_dict['true_lab']))
