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
from keras_vit.vit import ViT_B16
from keras_vit import preprocess_inputs as preprocess_inputs_vit


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


def predict_ViT(img):
    image_obj = img.resize((224, 224))
    model = ViT_B16(
        image_size=224,
        activation="softmax",
        pretrained=True,
        include_top=True,
        pretrained_top=True,
    )

    x = image.img_to_array(image_obj)
    x = np.expand_dims(x, axis=0)
    x = preprocess_inputs_vit(x)
    preds = model.predict(x, verbose=0)
    decoded_predictions = decode_predictions(preds, top=5)
    return decoded_predictions
