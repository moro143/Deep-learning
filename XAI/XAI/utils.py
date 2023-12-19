from models import predict_vgg16, predict_resnet, predict_efficientnet
import matplotlib.pyplot as plt
from xai_methods import lime
import traceback
from torch.nn.functional import softmax
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
import numpy as np


def check_num_classes(dataset, target_num_classes=30):
    class_labels = set()
    for sample in dataset:
        class_labels.add(sample["label"])
        if len(class_labels) == target_num_classes:
            return True

    return len(class_labels) == target_num_classes


def main_loop(dset, selected_indices):
    e = 0
    def fuc(images):
        model = EfficientNetB0(weights='imagenet')
        processed_images = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img_resized = img.resize((224, 224))
            x = keras_image.img_to_array(img_resized)
            x = preprocess_input(x)
            processed_images.append(x)
        processed_images = np.array(processed_images)
        preds = model.predict(processed_images, verbose=0)
        return preds
    for i in dset:
        try:
            img = i["image"]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

            ax1.imshow(i["image"])
            ax1.axis('off')

            # predicitons
            vgg_pred = str(predict_vgg16(img, selected_indices))
            resnet_pred = str(predict_resnet(img, selected_indices))
            efficientnet_pred = str(predict_efficientnet(img, selected_indices))

            prediction_text = (f"VGG Prediction: {vgg_pred}\n"
                               f"ResNet Prediction: {resnet_pred}\n"
                               f"EfficientNet Prediction: {efficientnet_pred}")

            ax2.text(0.5, 0.5, prediction_text, ha='center', va='center', fontsize=10, wrap=True)
            ax2.axis('off')

            # lime imgs
            lime_vgg = lime(img, fuc)
            print(lime_vgg)
            plt.imshow(lime_vgg)
            plt.axis('off')
            plt.show()
            plt.tight_layout()
            plt.show()

        except Exception as ex:
            print(traceback.format_exc())
            print(f"Error {e}")
            print(ex)
            e += 1
