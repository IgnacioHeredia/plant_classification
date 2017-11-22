# Webpage docs

This is a basic web app to implement image recognition with local files and urls. The `./webpage` folder is completelly self-contained and thus can be copied alone to your web server.

**Contents**

- `./model_files`: essential files for the image recognition test mode.
- `./webpage_files`: files for the webpage implementation and design.

## Preliminaries

First you have to copy the following files:

- `synsets.txt` to `./webpage/model_files/data` 
- your trained weights (`.npz`) to `./webpage/model_files/training_weights`
- your training info (`.json`) to  `./webpage/model_files/training_info`

You can additionally provide an  `info.txt` file to be put in `./webpage/model_files/data` with relevant information for each specie. 

## Launching the webpage

To launch the web execute the following:

```bash
cd ./webpage/webpage_files
export FLASK_APP=webpage_demo.py
python -m flask run
```
and it will start running at http://127.0.0.1:5000. To run the webpage in production mode you can pip-install the `gunicorn` module as an easy drop-in replacement. Once installed just run

```bash
cd ./webpage/webpage_files
gunicorn webpage_demo:app -b 0.0.0.0:80 --workers 4 --timeout 80 -k gevent
```

## Using the API

You can query your webpage also through an API. You have to make a POST request with the images belonging to your observation.

### Python snippets
Here are some Python snippets using the `requests` module.

**Classifying URLs**
```python
im_list = ['https://public-media.smithsonianmag.com/filer/89/47/8947cd5c-ac01-4c0e-891a-505517cc0663/istock-540753808.jpg', 
           'https://cdn.pixabay.com/photo/2014/04/10/11/24/red-rose-320868_960_720.jpg']

r = requests.post('http://127.0.0.1:5000/api', data={'mode':'url', 'url_list':im_list})
```

**Classifying local images**

```python
im_paths = ['/home/ignacio/image_recognition/data/demo-images/image1.jpg',
            '/home/ignacio/image_recognition/data/demo-images/image2.jpg']

im_names = [str(i) for i in range(len(im_paths))]
im_files = [open(f, 'rb') for f in im_paths]
file_dict = dict(zip(im_names, im_files))

r = requests.post('http://127.0.0.1:5000/api', data={'mode':'localfile'}, files=file_dict)
```

### CURL snippets

**Classifying URLs**
```bash
curl --data "mode=url&url_list=https://public-media.smithsonianmag.com/filer/89/47/8947cd5c-ac01-4c0e-891a-505517cc0663/istock-540753808.jpg&url_list=https://cdn.pixabay.com/photo/2014/04/10/11/24/red-rose-320868_960_720.jpg" http://127.0.0.1:5000/api
```

**Classifying local images**
```bash
curl --form mode=localfile --form 0=@/home/ignacio/image_recognition/data/demo-images/image1.jpg --form 1=@/home/ignacio/image_recognition/data/demo-images/image2.jpg http://deep.ifca.es/api
```

### Responses

A successful response should return a json, with the labels and their respective probabilities, like the following

```python
{u'pred_lab': [u'Rosa chinensis',
               u'Erythrina crista-galli',
               u'Tulipa agenensis',
               u'Gladiolus dubius',
               u'Spathodea campanulata'],
               
 u'pred_prob': [0.313213586807251,
                0.22123542428016663,
                0.037396140396595,
                0.033636994659900665,
                0.02710902690887451], 
                         
 u'info': [u'99 images in DB',
           u'35 images in DB',
           u'67 images in DB',
           u'49 images in DB',
           u'47 images in DB'],
           
 u'google_images_link': ['https://www.google.es/search?q=Rosa+chinensis&tbm=isch',
                         'https://www.google.es/search?q=Erythrina+crista-galli&tbm=isch',
                         'https://www.google.es/search?q=Tulipa+agenensis&tbm=isch',
                         'https://www.google.es/search?q=Gladiolus+dubius&tbm=isch',
                         'https://www.google.es/search?q=Spathodea+campanulata&tbm=isch'],
                       
 u'wikipedia_link': [u'https://en.wikipedia.org/wiki/Rosa_chinensis',
                     u'https://en.wikipedia.org/wiki/Erythrina_crista-galli',
                     u'https://en.wikipedia.org/wiki/Tulipa_agenensis',
                     u'https://en.wikipedia.org/wiki/Gladiolus_dubius',
                     u'https://en.wikipedia.org/wiki/Spathodea_campanulata'],
                     
 u'status': u'OK'
```
while unsuccessful responses will look like

```python
{u'Error_description': u"Some urls were not in image format. Check you didn't uploaded a preview of the image rather than the image itself.",
 u'Error_type': u'Url image format error',
 u'status': u'error'}
```
