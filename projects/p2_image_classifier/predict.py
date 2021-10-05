import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import json
tfds.disable_progress_bar()

import argparse

IMG_SIZE = 224

def parse_args():
    '''Defines python command line arguments and parses them'''
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()

    # first two positional arguments file and model paths
    parser.add_argument(dest='img_file', help="This is the image file location")
    parser.add_argument(dest='model_file', help="This is the model file loction")

    # Provide topK and category names flags
    parser.add_argument('-tk', '--topK', type=int, dest='top_k', default=5)
    parser.add_argument('-cn', '--category_names', type=str, dest='cat_names', default='label_map.json')
    
    # Parse and print the results
    args = parser.parse_args()
    
    return args

def load_class_names(class_file):
    '''Loads and returns the class name dictionary'''
    with open(class_file, 'r') as f:
        class_names = json.load(f)
    return class_names

def load_model(model_fname):
    '''Loads the model from file and prints summary'''
    # Load the Keras model
    reloaded_keras_model = tf.keras.models.load_model(model_fname,\
        custom_objects={'KerasLayer':hub.KerasLayer})

    # Print the model summary
    reloaded_keras_model.summary()
    return reloaded_keras_model


# The process_image function definition
def process_image(image):
    '''Resizes and normalizes the image for mobilenet'''
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image /= 255
    return image

def load_image(image_path):
    '''Test method for just loading image from file'''
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image, IMG_SIZE)

    return processed_test_image

def predict(img_path, model, topK):
    ''' Loads the image file and makes an inference on the image for label'''
    img_in = np.expand_dims(process_image(np.asarray(Image.open(img_path))), axis=0)
    pred = model.predict(img_in).squeeze()
    classes = (-pred).argsort()[:topK]
    probs = pred[classes]
    
    return img_in.squeeze(), probs, classes

def show_img(test_image):
    '''Test method to show the image in a modal window'''
    fig, (ax1) = plt.subplots(figsize=(5,5), ncols=1)
    ax1.imshow(test_image)
    ax1.set_title('Original Image')
    plt.tight_layout()
    plt.show()

def get_names(num_labels, class_names):
    '''Utility to get label strings from label number'''
    alpha_names = []
    for num in num_labels:
        alpha_names.append(class_names.get(str(num+1)))  # labels start at one, np.array at 0
    return alpha_names

def plot_pred(image, prob_top5, pred_top5, class_names):
    '''Matplotlib chart with image and probability barplot'''

    first_label = class_names.get(pred_top5[0]+1)
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(image, cmap = plt.cm.binary)
    ax1.axis('off')
    ax1.set_title(first_label)
    ax2.barh(np.arange(5), prob_top5)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels(get_names(pred_top5, class_names), size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()

def main():
    '''Inference driver calling the functions from jupyter notebook'''

    args = parse_args()
    print(args.img_file)
    print(args.model_file)
    print(args.top_k)
    print(args.cat_names)

    class_names = load_class_names(args.cat_names)
    model = load_model(args.model_file)
    #image = load_image(args.img_file)
    
    img, probs, classes = predict(args.img_file, model,args.top_k)
    #show_img(img)
    print(probs)
    print(get_names(classes+1,class_names)  # labels start at one, np.array at 0

    plot_pred(img,probs, classes, class_names)


if __name__ == "__main__":
    main()
