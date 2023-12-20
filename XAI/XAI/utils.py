import traceback
from models import ResNetPredict, EfficientNetPredict, Vgg16Predict
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries


def check_num_classes(dataset, target_num_classes=30):
    class_labels = set()
    for sample in dataset:
        class_labels.add(sample["label"])
        if len(class_labels) == target_num_classes:
            return True

    return len(class_labels) == target_num_classes


def main_loop(dset, selected_indices):
    e = 0
    models = {
        'ResNet': ResNetPredict(),
        'EfficientNet': EfficientNetPredict(),
        'Vgg': Vgg16Predict()
    }
    print(models['ResNet'].random_numbers)
    for i in dset:
        if i['label'] not in models['ResNet'].random_numbers:
            print('not', i)
            continue
        print('yes', i)
        try:
            img = i["image"]
            model_idx = 1
            predictions = {}
            for name, model in models.items():
                print(f'Starting - {name}')
                predictions[name] = model.predict_readable(img)
                print(model.predict_readable(img))
                try:
                    cam_image = model.cam(img)/255
                except:
                    cam_image=None
                try:
                    grad_cam_image = model.cam(img)/255
                except:
                    grad_cam_image=None
                overlayed_image, masked_image = model.lime_explain(img)
                num = 4
                
                try:
                    plt.subplot(len(models), num, model_idx)
                    plt.imshow(overlayed_image)
                    plt.title(f"{name} - LIME Explanation")
                    plt.axis("off")
                except:
                    print('Plot 1')

                try:
                    plt.subplot(len(models), num, model_idx + 1)
                    plt.imshow(masked_image)
                    plt.title(f"{name} - LIME Explanation (mask)")
                    plt.axis("off")
                except:
                    print('Plot 2')

                try:
                    plt.subplot(len(models), num, model_idx + 2)
                    plt.imshow(cam_image)
                    plt.title(f"{name} - CAM")
                    plt.axis("off")
                except:
                    print('Plot 3')

                try:
                    plt.subplot(len(models), num, model_idx + 3)
                    plt.imshow(grad_cam_image)
                    plt.title(f"{name} - Grad-CAM")
                    plt.axis("off")
                except:
                    print('Plot 4')

                model_idx += num
                print(f'Finished - {name}')
            plt.show()
   
        except Exception as ex:
            traceback.print_exc()
            print(f"Error {e}")
            print(ex)
            e += 1
