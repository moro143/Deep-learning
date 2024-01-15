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
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras import models
import random
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception, decode_predictions as decode_predictions_inception
import  shap

random.seed(2023)
rn = [random.randint(0, 999) for _ in range(30)]
class BasePredict:
    def __init__(self):
        self.model = None
        self.preprocess_input = None
        self.decode_predictions = None
        self.layer_cam = None
        self.random_numbers = rn
    def normalize_image(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.resize((224, 224))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return x

    def predict(self, img):
        img = self.normalize_image(img)
        img = self.preprocess_input(img)
        preds = self.model.predict(img, verbose=0)
        for i in range(len(preds[0])):
            if i not in self.random_numbers:
                preds[0][i] = 0.0
        return preds
    
    def predict_images(self, images):
        processed_images = []
        preds = []
        for img in images:
            x = self.normalize_image(img)
            x = self.preprocess_input(x)
            processed_images.append(x)
            y = self.predict(img)
            preds.append(self.predict(img)[0])
        processed_images = np.array(processed_images)
        return preds

    def lime_explain(self, img):
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(np.array(img), self.predict_images, num_samples=100, top_labels=1)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0])
        overlayed_image = mark_boundaries(temp, mask)
        masked_image = temp * mask[:, :, np.newaxis]
        return overlayed_image, masked_image, mask

    def grad_camv2(self, img):
        layer_name = self.layer_cam
        img_array = self.normalize_image(img)
        img_tensor = tf.convert_to_tensor(self.preprocess_input(img_array))

        model = self.model

        conv_layer = model.get_layer(layer_name)
        grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, tf.argmax(predictions[0])]

        grads = tape.gradient(loss, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, (img.width, img.height))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(np.uint8(img), 0.6, heatmap_color, 0.4, 0)

        return superimposed_img, heatmap

    def generate_saliency_map(self, img):
        img_array = self.normalize_image(img)
        img_tensor = tf.convert_to_tensor(self.preprocess_input(img_array))

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = self.model(img_tensor)
            loss = tf.reduce_max(predictions)

        grads = tape.gradient(loss, img_tensor)[0]

        # Normalize gradients
        grads = (grads - tf.reduce_min(grads)) / (tf.reduce_max(grads) - tf.reduce_min(grads))
        grads = tf.squeeze(grads).numpy()

        # Convert to grayscale
        grayscale_saliency = np.dot(grads, [0.2989, 0.5870, 0.1140])
        
        # Rescale to original image size
        saliency_map = cv2.resize(grayscale_saliency, (img.width, img.height))
        return saliency_map
    
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
        self.layer_cam = 'block5_conv3'

class InceptionV3Predict(BasePredict):
    def __init__(self):
        BasePredict.__init__(self)
        self.model = InceptionV3(weights="imagenet")
        self.preprocess_input = preprocess_input_inception
        self.decode_predictions = decode_predictions_inception
        self.layer_cam = 'mixed10'
    def normalize_image(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.resize((299, 299))
        x = keras_image.img_to_array(img)   # to zmieniales
        x = np.expand_dims(x, axis=0)
        return x
