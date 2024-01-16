import traceback
from models import ResNetPredict, EfficientNetPredict, Vgg16Predict, InceptionV3Predict
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv

def main_loop(dset):
    e = 0
    models = {
        #'InceptionV3': InceptionV3Predict(),
        #'ResNet': ResNetPredict(),
        #'EfficientNet': EfficientNetPredict(),
        'Vgg': Vgg16Predict()
    }
    c = 0
    for i in dset:
        if i['label'] not in models['Vgg'].random_numbers:
            continue
        try:
            img = i["image"]
            model_idx = 1
            predictions = {}
            plt.figure(figsize=(15, 10))
            for name, model in models.items():
                inv = model.use_innvestigate(img)
                print(f'Innvestigate: {inv}')
                # saliency_map = model.generate_saliency_map(img)
                # Display the original image and the saliency map side by side
                plt.figure(figsize=(12, 6))

                print(f'Starting - {name}')
                predictions[name] = model.predict_readable(img)
                print(predictions[name])
                # grad_cam_image, heatmap = model.grad_camv2(img)
                threshold = 0.1
                overlayed_image, masked_image_lime, mask = model.lime_explain(img)
                # masked_image_grad_cam = np.where(heatmap >= threshold, 1, 0)
                # print(masked_image_grad_cam)
                print('------------------')
                print(mask)
                print(f'Max masked_image_lime: {mask.max()}')

                # intersection = np.logical_and(mask, masked_image_grad_cam)
                # union = np.logical_or(mask, masked_image_grad_cam)
                # iou_score = np.sum(intersection) / np.sum(union)

                # print(f'IOU score: {iou_score}')
                num = 5

                plt.subplot(len(models), num, model_idx)
                plt.imshow(img)
                plt.title(f"Image")
                plt.axis("off")

                plt.subplot(len(models), num, model_idx + 1)
                plt.imshow(masked_image_lime)
                plt.title(f"LIME Explanation")
                plt.axis("off")

                #plt.subplot(len(models), num, model_idx + 2)
                #plt.imshow(masked_image_grad_cam)
                #plt.title(f"{name} - Binary Grad CAM")
                #plt.axis("off")

                #plt.subplot(len(models), num, model_idx + 3)
                #plt.imshow(grad_cam_image)
                #plt.title(f"{name} - Grad CAM")
                #plt.axis("off")
                #print(grad_cam_image.shape)
                print(inv.shape)
                plt.subplot(len(models), num, model_idx + 4)
                plt.imshow(inv[0])
                plt.title(f"{name} - Deep Taylor")
                plt.axis("off")
                model_idx += num
                print(f'Finished - {name}')
                continue
                heatmap = None # Delete this
                for i in range(10):
                    threshold_tmp = i*threshold
                    masked_image_grad_cam = np.where(heatmap >= threshold_tmp, 1, 0)
                    intersection = np.logical_and(mask, masked_image_grad_cam)
                    union = np.logical_or(mask, masked_image_grad_cam)
                    iou_score = np.sum(intersection) / np.sum(union)
                    data_to_save = {
                        'image': f'{c}.png',
                        'IOU score': iou_score,
                        'threshold': threshold_tmp,
                        'prediction': predictions[name][0][0][1],
                        'model': name,
                    }

                    csv_file_name = 'data.csv'

                    with open(csv_file_name, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=data_to_save.keys())
                        writer.writerow(data_to_save)
                
            plt.savefig(f'images/{c}.png')
            
            c += 1

        except Exception as ex:
            traceback.print_exc()
            print(f"Error {e}")
            print(ex)
            e += 1
