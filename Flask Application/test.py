from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
import SimpleITK as sitk
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os.path
import h5py
import s3fs

# Create Flask instance
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Set Max size of file as 100MB.

# Allow files with extension png, jpg, and jpeg
ALLOWED_EXTENSIONS = ['gz']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def unstack(a, axis=0):
    return np.moveaxis(a, axis, 0)


def read_image(filename):
    t1 = sitk.ReadImage(filename)
    t2 = sitk.GetArrayFromImage(t1)
    X = np.array(t2[(int(len(t2) / 2))])
    X = scipy.ndimage.zoom(X, 224 / 288)
    X = np.stack((X,) * 3, axis=-1)
    return X


def read_image1(filename):
    t1 = sitk.ReadImage(filename)
    t2 = sitk.GetArrayFromImage(t1)
    X = np.array(t2[(int(len(t2) / 2))])
    X = scipy.ndimage.zoom(X, 224 / 288)
    X = np.stack((X,) * 1, axis=-1)
    return X


def normalize(arr):
    arr = arr + abs(np.amin(arr))
    assert np.amax(arr) != 0
    arr = arr / np.amax(arr)
    return arr


def get_img_array(img_path, size):
    preprocess_input = keras.applications.xception.preprocess_input
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def load_model_from_s3(bucket_name, file_path):
    s3 = s3fs.S3FileSystem(anon=False, key="AKIATUCK53DTGZUOKPUN",
         secret= "23GYr4adyZJdlSZ8DH0/uit9yfe35YrdS8mWWr2w")
   # s3 = s3fs.S3FileSystem(anon=False, key="YOUR_AWS_ACCESS_KEY", secret="YOUR_AWS_SECRET_KEY")
    s3_file = s3.open(f"{bucket_name}/{file_path}", "rb")
    model = h5py.File(s3_file, "r")
    return model

def init():
    global model_mdd, model_szd
    # config = tf.compat.v1.ConfigProto(
    #     intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
    #     allow_soft_placement=True, device_count={'CPU': 1}
    # )
    # tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    s3 = s3fs.S3FileSystem(anon=False, key="AKIATUCK53DTGZUOKPUN",
         secret= "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")
    model_mdd = tf.keras.models.load_model(load_model_from_s3("trs-mri-bucket", "MOBILENET_depression.h5"), compile=False)
    model_szd = tf.keras.models.load_model(load_model_from_s3("trs-mri-bucket", "NOVEL_schizophrenia.h5"))


def get_class_activation_map(img, model):
    img = np.expand_dims(img, axis=0).astype(np.float32)
    predictions = model.predict(img)
    label_index = int(np.round(predictions[0][0]))

    class_weights = model.layers[-1].get_weights()[0]
    class_weights_winner = class_weights

    final_conv_layer = model.get_layer('resnet50').get_layer("conv5_block3_out")

    get_output = K.function([model.layers[0].input, model.get_layer('resnet50').layers[0].input],
                            [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img, img])

    conv_outputs = np.squeeze(conv_outputs)
    mat_for_mult = scipy.ndimage.zoom(conv_outputs, (27, 27, 1), order=1)
    final_output = np.dot(mat_for_mult.reshape((189 * 189, 2048)), class_weights_winner).reshape(189, 189)

    return final_output, label_index


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.85):
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)


def plot_class_activation_map(images, model):
    figure, axis = plt.subplots(1, 1)
    figure.set_size_inches(20, 20)

    current_y = 0
    current_x = 0

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

    for img in images:
        dx, dy = 10, 10
        grid_color = [0, 0, 0]
        img[:, ::dy, :] = grid_color
        img[::dx, :, :] = grid_color

        img = normalize(img)
        img = tf.squeeze(img)

        CAM, label = get_class_activation_map(img, model)

        coordinatesList = [[0, 0]]
        tx, ty = coordinatesList[0]

        axis.imshow(img, alpha=.6)
        axis.imshow(CAM, cmap='jet', alpha=0.5)

        class_label = "Input MRI Slice CAM output"
        axis.set_title(class_label, fontsize=60)

        if (current_x + 1) % 9 == 0:
            current_x = 0
            current_y += 1
        else:
            current_x += 1

    _, _, files = next(os.walk("static/results"))
    file_count = len(files)

    plt.savefig('static/results/resultMDD' + str(file_count) + '.png', bbox_inches='tight')

    return 'resultMDD' + str(file_count) + '.png'


def setHeatMap(file, path, model):
    img_array = get_img_array(path, size=(199, 137, 1))
    img1 = cv2.imread('static/gradInput/' + file)
    img1 = unstack(img1, axis=-1)

    model.layers[-1].activation = None
    img_array = np.expand_dims(img1[2], axis=0)

    heatmap = make_gradcam_heatmap(img_array, model, "conv2d_9")

    return heatmap


def plot_class_activation_map_SZD(images, model):
    figure, axis = plt.subplots(1, 1)
    figure.set_size_inches(20, 20)

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

    img_path = keras.utils.get_file(
        origin="file://" + os.path.abspath("") + "/static/gradInput/input.png"
    )
    heatmap = setHeatMap("input.png", img_path, model)
    save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.7)

    image = plt.imread("cam.jpg")
    coordinatesList = [[0, 0]]
    tx, ty = coordinatesList[0]

    axis.imshow(image)
    axis.set_title("Input MRI Slice CAM (GradCAM) output", fontsize=30)

    _, _, files = next(os.walk("static/results"))
    file_count = len(files)

    plt.savefig('static/results/resultSZD' + str(file_count) + '.png', bbox_inches='tight')

    return 'resultSZD' + str(file_count) + '.png'


@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path)

            predictions = model_mdd.predict(np.expand_dims(img, axis=0))
            label_index = int(np.round(predictions[0][0]))

            if label_index == 0:
                result = "Healthy MRI Slice"
            elif label_index == 1:
                result = "Major Depressive Disorder"

           # filename = plot_class_activation_map(np.array([img]), model_mdd)

            return render_template('mdd.html', result=result, filename=filename, css="show")

    return render_template('mdd.html')


@app.route("/predict1", methods=['POST'])
def predict1():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image1(file_path)

            predictions = model_szd.predict(np.expand_dims(img, axis=0))
            label_index = predictions[0][0]

            if label_index == 0.9999999:
                result = "Healthy TRS MRI Slice"
            else:
                result = "Schizophrenia"

            filename = plot_class_activation_map_SZD(1, model_szd)

            return render_template('szd.html', result=result, filename=filename, css="show")

    return render_template('szd.html')


@app.route("/mdd", methods=['GET', 'POST'])
def mddPage():
    return render_template('mdd.html')


@app.route("/szd", methods=['GET', 'POST'])
def szdPage():
    return render_template('szd.html')


@app.route("/ocd", methods=['GET', 'POST'])
def ocdPage():
    return render_template('ocd.html')


if __name__ == "__main__":
    init()
    app.run(debug=True)
