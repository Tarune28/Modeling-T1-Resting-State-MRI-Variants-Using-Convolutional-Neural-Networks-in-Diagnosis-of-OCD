#Import necessary libraries
from flask import Flask, render_template, request

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import os
import SimpleITK as sitk
import scipy
from tensorflow import keras
# Create flask instance
app = Flask(__name__)

#Set Max size of file as 10MB.
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['gz']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init():
    global graph
    graph = tf.compat.v1.get_default_graph()


def unstack(a, axis=0):
    return np.moveaxis(a, axis, 0)

def get_img_array(img_path, size):
    preprocess_input = keras.applications.xception.preprocess_input
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

# Function to load and prepare the image in right shape
def read_image(filename):
    t1 = sitk.ReadImage(filename)
    t2 = sitk.GetArrayFromImage(t1)
    print((int(len(t2)/2)))
    X = np.array(t2[(int(len(t2)/2))])
    X = scipy.ndimage.zoom(X, 224/288)
   # X = X.reshape(1, 244, 244)
    X = np.stack((X,)*3, axis=-1)
   # img = load_img(filename, grayscale=True, target_size=(28, 28))
    # Convert the image to array
    #img = img_to_array(img)
    # Reshape the image into a sample of 1 channel
   # img = img.reshape(1, 28, 28, 1)
    # Prepare it as pixel data
   # img = img.astype('float32')
   # img = img / 255.0
    return X

def read_image1(filename):
    t1 = sitk.ReadImage(filename)
    t2 = sitk.GetArrayFromImage(t1)
    print((int(len(t2)/2)))
    X = np.array(t2[(int(len(t2)/2))])
    X = scipy.ndimage.zoom(X, 224/288)
    X = np.stack((X,)*1, axis=-1)
   # X = X.reshape(1, 244, 244)

    return X

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path)
            # Predict the class of an image
            graph = tf.compat.v1.get_default_graph()
            with graph.as_default():
                model = load_model('RESNET_depression_resting_state_dataset_t1_2d.h5')

                img = np.expand_dims(img, axis=0).astype(np.float32)
                predictions = model.predict(img)
                label_index = int(np.round(predictions[0][0]))
             #   class_prediction = model1.predict_classes(img)
              #  print(class_prediction)

            #Map apparel category with the numerical class
            if label_index == 0:
              result = "Healthy MRI Slice"
            elif label_index == 1:
              result = "MDD"
  
            return render_template('mdd.html', result = result)
        

    return render_template('mdd.html')




@app.route("/predict1", methods = ['GET','POST'])
def predict1():
    preprocess_input = keras.applications.xception.preprocess_input
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image1(file_path)
            # Predict the class of an image
            graph = tf.compat.v1.get_default_graph()
            with graph.as_default():
                model = load_model('NOVEL_schizophrenia_resting_state_dataset_t1_2d.h5')
               
          
                img = np.expand_dims(img, axis=0)


                predictions = model.predict(img)
                print(predictions)
                label_index = predictions
             #   class_prediction = model1.predict_classes(img)
              #  print(class_prediction)

            #Map apparel category with the numerical class
            if label_index == 0.9999999:
              result = "Healthy TRS MRI Slice"
            elif label_index != 0.9999999:
              result = "SZD"
  
            return render_template('szd.html', result = result)
        

    return render_template('szd.html')

@app.route("/mdd", methods = ['GET','POST'])
def mddPage():
  return render_template('mdd.html')

@app.route("/szd", methods = ['GET','POST'])
def szdPage():
  return render_template('szd.html')

@app.route("/ocd", methods = ['GET','POST'])
def ocdPage():
  return render_template('ocd.html')


if __name__ == "__main__":
    init()
    app.run()
