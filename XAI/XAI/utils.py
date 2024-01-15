import traceback
from models import ResNetPredict, EfficientNetPredict, Vgg16Predict, InceptionV3Predict
import matplotlib.pyplot as plt
import numpy as np
import cv2

def main_loop(dset):
    e = 0
    models = {
        #'InceptionV3': InceptionV3Predict(),
        'ResNet': ResNetPredict(),
        #'EfficientNet': EfficientNetPredict(),
        #'Vgg': Vgg16Predict()
    }
    c = 0
    for i in dset:
        if i['label'] not in models['ResNet'].random_numbers:
            continue
        try:
            img = i["image"]
            model_idx = 1
            predictions = {}
            plt.figure(figsize=(15, 10))
            for name, model in models.items():
                print(f'Starting - {name}')
                predictions[name] = model.predict_readable(img)
                print(predictions[name])

                grad_cam_image, heatmap = model.grad_camv2(img)
                threshold = heatmap.max() / 4
                overlayed_image, masked_image_lime, mask = model.lime_explain(img)
                masked_image_grad_cam = np.where(heatmap >= threshold, 1, 0)
                print(masked_image_grad_cam)
                print('------------------')
                print(mask)
                print(f'Max masked_image_lime: {mask.max()}')


                intersection = np.logical_and(mask, masked_image_grad_cam)
                union = np.logical_or(mask, masked_image_grad_cam)
                iou_score = np.sum(intersection) / np.sum(union)

                print(f'IOU score: {iou_score}')
                num = 4

                plt.subplot(len(models), num, model_idx)
                plt.imshow(img)
                plt.title(f"Image")
                plt.axis("off")

                plt.subplot(len(models), num, model_idx + 1)
                plt.imshow(masked_image_lime)
                plt.title(f"LIME Explanation")
                plt.axis("off")

                plt.subplot(len(models), num, model_idx + 2)
                #plt.imshow(grad_cam_image)
                plt.imshow(masked_image_grad_cam)
                plt.title(f"{name} - Grad CAM")
                plt.axis("off")

                plt.subplot(len(models), num, model_idx + 3)
                plt.imshow(grad_cam_image)
                plt.title(f"{name} - Grad CAM")
                plt.axis("off")
                model_idx += num
                print(f'Finished - {name}')
            plt.savefig(f'images/{c}.png')
            c += 1

        except Exception as ex:
            traceback.print_exc()
            print(f"Error {e}")
            print(ex)
            e += 1
