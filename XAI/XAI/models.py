import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input as preprocess_input_resnet,
)
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0,
    preprocess_input as preprocess_input_efficientnet,
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions


def predict_resnet(img):
    image_obj = img.resize((224, 224))
    model = ResNet50(weights="imagenet")
    x = image.img_to_array(image_obj)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input_resnet(x)

    preds = model.predict(x, verbose=0)
    return decode_predictions(preds)


def predict_efficientnet(img):
    image_obj = img.resize((224, 224))
    model = EfficientNetB0(weights="imagenet")
    x = image.img_to_array(image_obj)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input_efficientnet(x)

    preds = model.predict(x, verbose=0)
    return decode_predictions(preds)


def predict_vgg16(img):
    image_obj = img.resize((224, 224))
    model = VGG16(weights='imagenet')
    x = image.img_to_array(image_obj)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)
    return decode_predictions(preds)
