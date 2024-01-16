import traceback
from models import ResNetPredict, EfficientNetPredict, Vgg16Predict, InceptionV3Predict
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv

def identify_important_pixels(relevance_scores, threshold=0.1):
    """
    Identify important pixels based on relevance scores and a percentile threshold.
    
    :param relevance_scores: A numpy array of relevance scores from Deep Taylor Decomposition.
    :param percentile_threshold: The percentile to use as the threshold for important pixels (default is 95).
    :return: A binary numpy array where 1 indicates an important pixel, and 0 indicates a non-important pixel.
    """
    # Normalize the relevance scores to be between 0 and 1
    normalized_scores = relevance_scores - np.min(relevance_scores)
    if np.max(normalized_scores) != 0:
        normalized_scores = normalized_scores / np.max(normalized_scores)

    # Identify pixels that meet or exceed the threshold
    important_pixels = np.where(normalized_scores >= threshold, 1, 0)

    return important_pixels.astype(int)


def main_loop(dset):
    e = 0
    models = {
        # 'InceptionV3': InceptionV3Predict(),
        # 'ResNet': ResNetPredict(),
        # 'EfficientNet': EfficientNetPredict(),
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
                important_pixels = identify_important_pixels(inv)

                print(f'Starting - {name}')
                predictions[name] = model.predict_readable(img)
                print(predictions[name])
                grad_cam_image, heatmap = model.grad_camv2(img)
                threshold = 0.6
                overlayed_image, masked_image_lime, mask = model.lime_explain(img)
                masked_image_grad_cam = np.where(heatmap >= threshold, 1, 0)
                print(mask.shape, important_pixels.shape, masked_image_grad_cam.shape)
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
                    'prediction': predictions[name][0][0][1],
                    'model': name,
                }

                csv_file_name = 'data.csv'

                with open(csv_file_name, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=data_to_save.keys())
                    writer.writerow(data_to_save)

                num = 6

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
                plt.title(f"{name} - Binary Grad CAM")
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
                plt.title(f"{name} - Deep Taylor Binary")
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
