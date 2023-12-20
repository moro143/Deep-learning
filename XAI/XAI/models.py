from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import (
    ResNet50,
    EfficientNetB0,
    VGG16,
    imagenet_utils,
)
from tensorflow.keras.applications.resnet50 import (
    preprocess_input as preprocess_input_resnet,
    decode_predictions as decode_predictions_resnet
)
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as preprocess_input_efficientnet,
    decode_predictions as decode_predictions_efficientnet
)
from keras.applications.vgg16 import preprocess_input, decode_predictions
import traceback
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras import models
import shap
import random

random.seed(98)
rn = [random.randint(0, 999) for _ in range(30)]
class BasePredict:
    def __init__(self):
        self.model = None
        self.preprocess_input = None
        self.decode_predictions = None
        self.layer_cam = None
        self.random_numbers = rn

    def predict(self, img):
        img = img.resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)
        preds = self.model.predict(x, verbose=0)
        for i in range(len(preds[0])):
            if i not in self.random_numbers:
                preds[0][i] = 0.0
            else:
                print(i)
        return preds
    
    def predict_images(self, images):
        processed_images = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img_resized = img.resize((224, 224))
            x = keras_image.img_to_array(img_resized)
            x = self.preprocess_input(x)
            processed_images.append(x)
        processed_images = np.array(processed_images)
        preds = self.model.predict(processed_images, verbose=0)
        return preds

    def lime_explain(self, img, top_labels=5):
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(np.array(img), 
                                                 self.predict_images)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0])
        overlayed_image = mark_boundaries(temp, mask)
        masked_image = temp * mask[:, :, np.newaxis]
        return overlayed_image, masked_image

    def cam(self, img):
        img_resized = img.resize((224, 224))
        x = keras_image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)

        model_output = self.model.predict(x)
        last_conv_layer = self.model.get_layer(self.layer_cam)
        last_conv_model = models.Model(self.model.inputs, last_conv_layer.output)
        last_conv_output = last_conv_model.predict(x)

        output_weights = self.model.get_layer('predictions').get_weights()[0]
        class_idx = np.argmax(model_output[0])
        class_output_weights = output_weights[:, class_idx]

        cam_output = np.dot(last_conv_output[0], class_output_weights)
        cam_output = cv2.resize(cam_output, (224, 224))
        cam_output = np.maximum(cam_output, 0)
        heatmap = cam_output / np.max(cam_output)

        original_img = np.array(img_resized)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + original_img
        return superimposed_img

    def grad_cam(self, img, layer_name=None):
        if layer_name is None:
            layer_name = self.layer_cam

        img_resized = img.resize((224, 224))
        x = keras_image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)

        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(layer_name).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        conv_outputs = conv_outputs @ pooled_grads[..., tf.newaxis]
        conv_outputs = tf.squeeze(conv_outputs)

        heatmap = tf.maximum(conv_outputs, 0) / tf.math.reduce_max(conv_outputs)

        heatmap = np.uint8(255 * heatmap.numpy())
        heatmap = cv2.resize(heatmap, (img_resized.width, img_resized.height))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = heatmap * 0.4 + np.array(img_resized)
        superimposed_img = np.uint8(superimposed_img)

        return superimposed_img
    
    def predict_readable(self, img):
        return self.decode_predictions(self.predict(img))

class ResNetPredict(BasePredict):
    def __init__(self):
        BasePredict.__init__(self)
        self.model = ResNet50(weights="imagenet")
        self.preprocess_input = preprocess_input_resnet
        self.decode_predictions = decode_predictions_resnet
        self.layer_cam = 'conv5_block3_out'

class EfficientNetPredict(BasePredict):
    def __init__(self):
        BasePredict.__init__(self)
        self.model = EfficientNetB0(weights="imagenet")
        self.preprocess_input = preprocess_input_efficientnet
        self.decode_predictions = decode_predictions_efficientnet
        self.layer_cam = 'top_conv'

class Vgg16Predict(BasePredict):
    def __init__(self):
        BasePredict.__init__(self)
        self.model = VGG16(weights="imagenet")
        self.preprocess_input = preprocess_input
        self.decode_predictions = decode_predictions
        self.layer_cam = 'block2_conv2'

