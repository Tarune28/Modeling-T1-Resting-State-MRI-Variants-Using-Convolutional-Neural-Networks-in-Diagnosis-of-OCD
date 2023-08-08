#Import necessary libraries
from flask import Flask, render_template, request

import numpy as np
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import os
import SimpleITK as sitk
import scipy
from tensorflow import keras
import matplotlib.pyplot as plt 
from PIL import Image
import matplotlib.cm as cm
import matplotlib
matplotlib.use('agg')
import os.path
from memory_profiler import memory_usage
from time import sleep


# Create flask instance
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

#Set Max size of file as 10MB.
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['gz']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init():
  pass

# def unstack(a, axis=0):
#     return np.moveaxis(a, axis, 0)

# def get_img_array(img_path, size):
#     preprocess_input = keras.applications.xception.preprocess_input
#     img = keras.preprocessing.image.load_img(img_path, target_size=size)
#     array = keras.preprocessing.image.img_to_array(img)
#     array = np.expand_dims(array, axis=0)
#     return array

# # Function to load and prepare the image in right shape
# def read_image(filename):
#     t1 = sitk.ReadImage(filename)
#     t2 = sitk.GetArrayFromImage(t1)
#     print((int(len(t2)/2)))
#     X = np.array(t2[(int(len(t2)/2))])
#     X = scipy.ndimage.zoom(X, 224/288)
#    # X = X.reshape(1, 244, 244)
#     X = np.stack((X,)*3, axis=-1)
#    # img = load_img(filename, grayscale=True, target_size=(28, 28))
#     # Convert the image to array
#     #img = img_to_array(img)
#     # Reshape the image into a sample of 1 channel
#    # img = img.reshape(1, 28, 28, 1)
#     # Prepare it as pixel data
#    # img = img.astype('float32')
#    # img = img / 255.0
#     return X


# def read_image1(filename):
#     t1 = sitk.ReadImage(filename)
#     t2 = sitk.GetArrayFromImage(t1)
#     print((int(len(t2)/2)))
#     X = np.array(t2[(int(len(t2)/2))])
#     X = scipy.ndimage.zoom(X, 224/288)
#     X = np.stack((X,)*1, axis=-1)
#    # X = X.reshape(1, 244, 244)

#     return X

# def normalize(arr):
#     arr = arr + abs(np.amin(arr))
#     assert np.amax(arr) != 0
#     arr = arr / np.amax(arr)
#     return arr

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('home.html')

# def get_class_activation_map(img, MODEL_PATH):

#     model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
#     img = np.expand_dims(img, axis=0).astype(np.float32)
    
  
#     predictions = model.predict(img)
#     label_index = int(np.round(predictions[0][0]))

#     class_weights = model.layers[-1].get_weights()[0]
#     class_weights_winner = class_weights
#     #[:, label_index]

#     # get the final conv layer
#     final_conv_layer = model.get_layer('mobilenetv2_1.00_224').get_layer("Conv_1_bn")
#     final_conv_layer

#     # create a function to fetch the final conv layer output maps (should be shape (1, 7, 7, 2048)) 
#     get_output = K.function([model.layers[0].input, model.get_layer('mobilenetv2_1.00_224').layers[0].input],[final_conv_layer.output, model.layers[-1].output])
#     [conv_outputs, predictions] = get_output([img, img])

#     conv_outputs = np.squeeze(conv_outputs)
#     mat_for_mult = scipy.ndimage.zoom(conv_outputs, (32, 32, 1), order=1) 
#     final_output = np.dot(mat_for_mult.reshape((224*224, 1280)), class_weights_winner).reshape(224,224) # dim: 224 x 224

#     return final_output, label_index
   

# def plot_class_activation_map(images, MODEL_PATH):

  
#     figure, axis = plt.subplots(1,1)

#     figure.set_size_inches(20, 20)

#     current_y = 0
#     current_x = 0
    
#     plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

#     i = 0

#     for img in images:
        
#         dx, dy = 10,10

#         grid_color = [0,0,0]

#         img[:,::dy,:] = grid_color
#         img[::dx,:,:] = grid_color

#         img = normalize(img)

#         img = tf.squeeze(img)
        
#         CAM, label = get_class_activation_map(img, MODEL_PATH)
        

#         coordinatesList = [[0, 0]]
#         tx, ty = coordinatesList[0]
       
        
        
#         axis.imshow(img, alpha=.6)
#         axis.imshow(CAM, cmap='jet', alpha=0.5)
              
        
#         class_label = "Input MRI Slice CAM output"
#         axis.set_title(class_label, fontsize=60)

#         if (current_x+1)%9 == 0:
#             current_x = 0
#             current_y +=1
#         else:
#             current_x += 1
#         i+=1

#         #print("MRI number" + i + "produced")
#         _, _, files = next(os.walk("static/results"))
#         file_count = len(files)

#     plt.savefig('static/results/resultMDD'+str(file_count)+'.png', bbox_inches='tight')

#     return 'resultMDD'+str(file_count)+'.png'

# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
  
#     grad_model = tf.keras.models.Model(
#         [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         last_conv_layer_output, preds = grad_model(img_array)
#         if pred_index is None:
#             pred_index = tf.argmax(preds[0])
#         class_channel = preds[:, pred_index]

#     grads = tape.gradient(class_channel, last_conv_layer_output)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     last_conv_layer_output = last_conv_layer_output[0]
#     heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy()

# def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.85):

#     img = keras.preprocessing.image.load_img(img_path)
#     img = keras.preprocessing.image.img_to_array(img)

#     heatmap = np.uint8(255 * heatmap)

#     jet = cm.get_cmap("jet")

#     jet_colors = jet(np.arange(256))[:, :3]
#     jet_heatmap = jet_colors[heatmap]

#     jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
#     jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
#     jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

#     superimposed_img = jet_heatmap * alpha + img
#     superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

#     superimposed_img.save(cam_path)

# def setHeatMap(file, path, MODEL_PATH):
#     preprocess_input = keras.applications.xception.preprocess_input
#     img_array = preprocess_input(get_img_array(path, size=(199, 137, 1)))
#     print(file + " " + path)
#     img1 = np.asarray(Image.open('static/gradInput/' + file))
#     #test = cv2.imread('static/gradInput/' + file)

#     img1 = np.stack((img1,)*3, axis=-1)
#     print(img1)
#     print(img1.shape)



#     img1 = unstack(img1, axis=-1)

#     # img_array = tf.squeeze(img_array)
 
#     # img_array = np.expand_dims(img1, axis=-1)/Users/taruneswar/Documents/General Development/Modeling-T1-Resting-State-MRI-Variants-Using-Convolutional-Neural-Networks-in-Diagnosis-of-OCD/Prediction_Flask_Web_App/web-app/models/NOVEL_schizophrenia_resting_state_dataset_t1_2d.h5
#     model = tf.keras.models.load_model(MODEL_PATH)
#     model.layers[-1].activation = None
#     img_array = np.expand_dims(img1[2], axis=0)

#     # preds = model.predict(img_array)
#     heatmap = make_gradcam_heatmap(img_array, model, "conv2d_9")

#     return heatmap

# def plot_class_activation_map_SZD(images, MODEL_PATH):

#     figure, axis = plt.subplots(1,1)

#     figure.set_size_inches(20,20)

#     current_y = 0
#     current_x = 0
    
#     plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

    

#     for i in range(images):

#         print(os.path.abspath("input.png"))
#         img_path = keras.utils.get_file(
#         origin="file://" + os.path.abspath("") + "/static/gradInput/input.png"
#     )
#         heatmap = setHeatMap("input.png", img_path, MODEL_PATH)

#         save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.7)
        
#         image = plt.imread("cam.jpg")
#         coordinatesList = [[0, 0]]
#         tx, ty = coordinatesList[0]
        
       
#         axis.imshow(image)
#         axis.set_title("Input MRI Slice CAM (GradCAM) output", fontsize=30)
    
#         # class_label = "Patient " + str(i+1)
#         # axis[current_y, current_x].set_title(class_label)

       

#         #print("MRI number" + i + "produced")

    
#     _, _, files = next(os.walk("static/results"))
#     file_count = len(files)

#     plt.savefig('static/results/resultSZD' + str(file_count) + '.png', bbox_inches='tight')

#     return 'resultSZD' + str(file_count) + '.png'

# @app.route("/predict", methods = ['GET','POST'])
# def predict():
#     # config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
#     #                     allow_soft_placement=True, device_count = {'CPU': 1})
#     # tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config));

   
#     f = "models/MOBILENET_depression_resting_state_dataset_t1_2d.h5"
#     if request.method == 'POST':
#         file = request.files['file']
#         if file and allowed_file(file.filename):
#             filename = file.filename
#             file_path = os.path.join('static/images', filename)
#             file.save(file_path)
#             img = read_image(file_path)
#             # Predict the class of an image
#             # graph = tf.compat.v1.get_default_graph()

#             # with graph.as_default():
#             model = load_model(f, compile=False)

#             img = np.expand_dims(img, axis=0).astype(np.float32)
#             predictions = model.predict(img)
#             label_index = int(np.round(predictions[0][0]))
#             print(predictions[0][0])
#              #   class_prediction = model1.predict_classes(img)
#               #  print(class_prediction)



#             #Map apparel category with the numerical class
#             if label_index == 0:
#               result = "Healthy MRI Slice"
#             elif label_index == 1:
#               result = "Major Depressive Disorder"

#             filename = plot_class_activation_map(img, f)

            
  
#             return render_template('mdd.html', result = result, filename = filename, css="show")
        

#     return render_template('mdd.html')




# @app.route("/predict1", methods = ['GET','POST'])
# def predict1():

#     # config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
#     #                     allow_soft_placement=True, device_count = {'CPU': 1})
#     # tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
#     f = "models/szd.h5"
#     if request.method == 'POST':
#         file = request.files['file']
#         if file and allowed_file(file.filename):
#             filename = file.filename
#             file_path = os.path.join('static/images', filename)
#             file.save(file_path)
#             X = []
#             t1 = sitk.ReadImage(file_path)
#             t2 = sitk.GetArrayFromImage(t1)
#             t3 = []
#             for j in range(107,147):
#                 result = scipy.ndimage.zoom(t2[j], 224/288)
#                 t3.append(result)
#             X.append(t3)

#             img = read_image1(file_path)
#             im = Image.fromarray(X[0][20])
#             im = im.convert("L")
#             im.save("static/gradInput/input.png")
#             graph = tf.compat.v1.get_default_graph()

            
#             with graph.as_default():
#                 model = load_model(f)
#                 img = np.expand_dims(img, axis=0)
#                 predictions = model.predict(img)
#                 label_index = predictions
#             if label_index == 0.9999999:
#               result = "Healthy TRS MRI Slice"
#             elif label_index != 0.9999999:
#               result = "Schizophrenia"

#             filename = plot_class_activation_map_SZD(1, f)
  
#             return render_template('szd.html', result = result, filename = filename, css="show")
        

#     return render_template('szd.html')

# @app.route("/ResnetMDD", methods = ['GET','POST'])
# def ResnetMDD():


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
    app.run(debug=True)

mem_usage = memory_usage(f)
print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
print('Maximum memory usage: %s' % max(mem_usage))