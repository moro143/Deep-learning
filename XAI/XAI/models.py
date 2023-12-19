from PIL import Image
import cv2
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
import traceback


resnet_model = ResNet50(weights="imagenet")
efficientnet_model = EfficientNetB0(weights="imagenet")
vgg16_model = VGG16(weights="imagenet")


def preprocess_image(img):
    """Resize and preprocess the image for model prediction."""
    # Check if the image is a NumPy array
    if isinstance(img, np.ndarray):
        # Resize using cv2 (OpenCV) if the input is a NumPy array
        resized_img = cv2.resize(img, (224, 224))
    else:
        # If it's a PIL Image, use the existing method
        image_obj = img.resize((224, 224))
        resized_img = image.img_to_array(image_obj)

    return np.expand_dims(resized_img, axis=0)



def predict_with_model(model, preprocess_input, img, selected_indices, xai=True):
    """Generic function to predict image using a specified model."""
    try:
        x = preprocess_image(img)
        x = preprocess_input(x)
        preds = model.predict(x, verbose=0)

       # Zero out predictions for classes not in the selected indices
        for i in range(len(preds[0])):
            if i not in selected_indices:
                preds[0][i] = 0.0

        # Normalize the probabilities
        total_prob = sum(preds[0])
        if total_prob > 0:
            preds[0] = [p / total_prob for p in preds[0]]

        decoded = imagenet_utils.decode_predictions(preds)
        if xai:
            return preds
        return decoded
    except Exception as e:
        print(traceback.format_exc())
        print(f"An error occurred: {e}")
        return []


def predict_resnet(img, selected_indices):
    """Predict image using ResNet50 model."""
    return predict_with_model(resnet_model, preprocess_input_resnet, img, selected_indices)


def predict_efficientnet(img, selected_indices):
    """Predict image using EfficientNetB0 model."""
    return predict_with_model(efficientnet_model, preprocess_input_efficientnet, img, selected_indices)


def predict_vgg16(img, selected_indices):
    """Predict image using VGG16 model."""
    return predict_with_model(vgg16_model, preprocess_input, img, selected_indices)
