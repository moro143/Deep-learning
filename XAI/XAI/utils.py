import traceback
from models import ResNetPredict, EfficientNetPredict, Vgg16Predict, InceptionV3Predict
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv

def identify_important_pixels(relevance_scores, threshold=0.1):
    normalized_scores = relevance_scores - np.min(relevance_scores)
    if np.max(normalized_scores) != 0:
        normalized_scores = normalized_scores / np.max(normalized_scores)

    important_pixels = np.where(normalized_scores >= threshold, 1, 0)

    return important_pixels.astype(int)


def main_loop(dset, thresholds=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
    e = 0
    models = {
        'Vgg': Vgg16Predict()
    }
    
    label_count = {}

    c = 0
    for i in dset:
        if i['label'] not in models['Vgg'].random_numbers:
            continue

        label_count[i['label']] = label_count.get(i['label'], 0) + 1

        if label_count[i['label']] > 3:
            continue
        
        try:
            img = i["image"]
            model_idx = 1
            predictions = {}
            plt.figure(figsize=(15, 10))
            for name, model in models.items():
                inv = model.use_innvestigate(img)
                grad_cam_image, heatmap = model.grad_camv2(img)
                overlayed_image, masked_image_lime, mask = model.lime_explain(img)
                predictions[name] = model.predict_readable(img)

                for dt_threshold in thresholds:
                    for gc_threshold in thresholds:
                        important_pixels = identify_important_pixels(inv, threshold=dt_threshold)
                        masked_image_grad_cam = np.where(heatmap >= gc_threshold, 1, 0)

                        intersectionLG = np.logical_and(mask, masked_image_grad_cam)
                        intersectionLD = np.logical_and(mask, important_pixels)
                        intersectionDG = np.logical_and(masked_image_grad_cam, important_pixels[0])
                        
                        unionLG = np.logical_or(mask, masked_image_grad_cam)
                        unionLD = np.logical_or(mask, important_pixels)
                        unionDG = np.logical_or(important_pixels, masked_image_grad_cam)
                        
                        iou_scoreLG = np.sum(intersectionLG) / np.sum(unionLG)
                        iou_scoreLD = np.sum(intersectionLD) / np.sum(unionLD)
                        iou_scoreDG = np.sum(intersectionDG) / np.sum(unionDG)

                        data_to_save = {
                            'image': f'{c}.png',
                            'IOU score Lime vs GradCAM': iou_scoreLG,
                            'IOU score Lime vs Deep_Taylor': iou_scoreLD,
                            'IOU score Deep_Taylor vs GradCAM': iou_scoreDG,
                            'threshold_DT': dt_threshold,
                            'threshold_gc': gc_threshold,
                            'prediction': predictions[name][0][0][1],
                            'model': name,
                        }

                        csv_file_name = 'data.csv'

                        with open(csv_file_name, 'a', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=data_to_save.keys())
                            writer.writerow(data_to_save)

                num = 6
                thgd = 0.5
                thdt = 0.2
                masked_image_grad_cam = np.where(heatmap >= thgd, 1, 0)
                important_pixels = identify_important_pixels(inv, threshold=thdt)
                plt.subplot(len(models), num, model_idx)
                plt.imshow(img)
                plt.title(f"Image")
                plt.axis("off")

                plt.subplot(len(models), num, model_idx + 1)
                plt.imshow(masked_image_lime)
                plt.title(f"LIME Explanation")
                plt.axis("off")

                plt.subplot(len(models), num, model_idx + 2)
                plt.imshow(masked_image_grad_cam)
                plt.title(f"{name} - Binary Grad CAM - th = {thgd}")
                plt.axis("off")

                plt.subplot(len(models), num, model_idx + 3)
                plt.imshow(grad_cam_image)
                plt.title(f"{name} - Grad CAM")
                plt.axis("off")
                
                plt.subplot(len(models), num, model_idx + 4)
                plt.imshow(inv)
                plt.title(f"{name} - Deep Taylor")
                plt.axis("off")

                plt.subplot(len(models), num, model_idx + 5)
                plt.imshow(important_pixels)
                plt.title(f"{name} - Deep Taylor Binary - th = {thdt}")
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