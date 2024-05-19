import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.utils import load_img, img_to_array, array_to_img
from keras.models import Model
import tensorflow as tf
import matplotlib as mpl
import cv2

def evaluate_model(model, model_name, X_val, y_val):
    y_pred = model.predict(X_val)
    y_pred = [np.argmax(y) for y in y_pred]
    # Calculate metrics
    accuracy = round(accuracy_score(y_val, y_pred),4)
    precision = round(precision_score(y_val, y_pred, average='weighted'),4)
    recall = round(recall_score(y_val, y_pred, average='weighted'),4)
    f1 = round(f1_score(y_val, y_pred, average='weighted'),4)
    # Append results to DataFrame
    return [model_name, accuracy, precision, recall, f1]

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4):
    # Load the original image
    img = load_img(img_path)
    img = img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)

    # Display Grad CAM
    return superimposed_img

def make_generators(datagen, data, x_col, y_col, img_size, batch_size, class_mode):

    train_generator = datagen.flow_from_dataframe(
        dataframe=data,
        directory=None,
        x_col=x_col,
        y_col=y_col,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset='training')

    # Create the validation generator
    validation_generator = datagen.flow_from_dataframe(
        dataframe=data,
        directory=None,
        x_col=x_col,
        y_col=y_col,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset='validation')

    return train_generator, validation_generator

def get_img_paths(generator):
    directory = generator.directory
    filenames = generator.filenames
    return [os.path.join(directory, filename) for filename in filenames]

def get_alterations_and_difficulties(paths):
    alterations = []
    difficulties = []

    for path in tqdm(paths):
        if "real" in path:
            alteration = "Nope"
            difficulty = "Nope"
        else:
            alteration = path.split('__')[1].split('_')[-2]
            difficulty = path.split('_')[-1].split('.')[0]
        
        alterations.append(alteration)
        difficulties.append(difficulty)

    return alterations, difficulties
def get_val_xy(paths, label_to_class):
    X_val = []
    y_val = []

    for path in tqdm(paths):
        img = cv2.imread(path)
        img = cv2.resize(img, (192, 206))
        img = img / 255.0
        X_val.append(img)

        # if any of class in the path, append the corresponding label to y_val
        for label, clas in label_to_class.items():
            if clas in path:
                y_val.append(label)
                break
            elif 'real' in path:
                y_val.append(1) 
                break

    return np.array(X_val), np.array(y_val)


def get_vals(validation_generator):
    X_val = []
    y_val = []

    # Iterate over the validation generator to retrieve Xs and Ys
    for i in tqdm(range(len(validation_generator))):
        # Get a batch of data from the generator
        batch_X, batch_y = next(validation_generator)
        # Append the batch to the Xs and Ys lists
        X_val.append(batch_X)
        y_val.append(batch_y)

    # Concatenate the lists to form numpy arrays
    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)
    return X_val, y_val