import os
import cv2
import json
import math
import numpy as np
from PIL import Image

import tensorflow as tf

from keras import layers
from keras.optimizers import Adam
from keras.models import Sequential
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


"""
Pretrained model accepts image resolution of 224x224, 
therefore image is resized to this resolution
"""
def preprocess_image(image_path, desired_size=224):
    img = Image.open(image_path)
    img = img.resize((desired_size,) * 2, resample=Image.LANCZOS)

    return img

"""
Function to build the model for inference on the image
"""
def build_model():
    """
    Pretrained model use Densenet as the base model in specific DenseNet121 is used
    Model weights are saved in file DenseNet-BC-121-32-no-top.h5
    """
    densenet = DenseNet121(
        weights="models/pretrained/DenseNet-BC-121-32-no-top.h5",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    """
    On top of Densenet average pooling layer, dropout layer and 
    finally dense layer with 5 classsed with sigmoid is used  
    """
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation="sigmoid"))


    """
    model parameters used in training:
    optimizer : adam
    learning rate : 10e-5
    """
    model.compile(
        loss="binary_crossentropy", optimizer=Adam(lr=0.00005), metrics=["accuracy"]
    )

    return model

"""
Image are classified into 5 categories based on severity using the pretrained model
Return the maximum probability class for given image
"""
def classify_image(img):
    # Build model used for classification
    # Load weights from pretrained model
    model = build_model()
    model.load_weights("models/pretrained/model.h5")

    # Create preprocessed image to be evaluated and predict its class
    x_val = np.empty((1, 224, 224, 3), dtype=np.uint8)
    x_val[0, :, :, :] = preprocess_image(img)
    y_val_pred = model.predict(x_val)

    # Return the maximum probability class
    return np.argmax(np.squeeze(y_val_pred[0]))
