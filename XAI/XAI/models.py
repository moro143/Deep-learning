import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import (
    ResNet50,
    EfficientNetB0,
    VGG16,
    imagenet_utils,
)
from tensorflow.keras.applications.resnet50 import (
    preprocess_input as preprocess_input_resnet,
)
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as preprocess_input_efficientnet,
)
from keras.applications.vgg16 import preprocess_input


resnet_model = ResNet50(weights="imagenet")
efficientnet_model = EfficientNetB0(weights="imagenet")
vgg16_model = VGG16(weights="imagenet")


def preprocess_image(img):
    """Resize and preprocess the image for model prediction."""
    image_obj = img.resize((224, 224))
    x = image.img_to_array(image_obj)
    return np.expand_dims(x, axis=0)


def predict_with_model(model, preprocess_input, img):
    """Generic function to predict image using a specified model."""
    try:
        x = preprocess_image(img)
        x = preprocess_input(x)
        preds = model.predict(x, verbose=0)
        return imagenet_utils.decode_predictions(preds), preds
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def predict_resnet(img):
    """Predict image using ResNet50 model."""
    return predict_with_model(resnet_model, preprocess_input_resnet, img)


def predict_efficientnet(img):
    """Predict image using EfficientNetB0 model."""
    return predict_with_model(efficientnet_model, preprocess_input_efficientnet, img)


def predict_vgg16(img):
    """Predict image using VGG16 model."""
    return predict_with_model(vgg16_model, preprocess_input, img)
