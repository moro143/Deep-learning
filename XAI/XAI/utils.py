import traceback
from models import ResNetPredict, EfficientNetPredict, Vgg16Predict, InceptionV3Predict
import matplotlib.pyplot as plt

def main_loop(dset):
    e = 0
    models = {
        'InceptionV3': InceptionV3Predict(),
        'ResNet': ResNetPredict(),
        'EfficientNet': EfficientNetPredict(),
        'Vgg': Vgg16Predict()
    }
    c = 0
    for i in dset:
        if i['label'] not in models['ResNet'].random_numbers:
            continue
        try:
            img = i["image"]
            model_idx = 1
            predictions = {}
            plt.figure(figsize=(15,10))
            for name, model in models.items():
                print(f'Starting - {name}')
                predictions[name] = model.predict_readable(img)
                model.normalize_image(img)
                print(model.predict_readable(img))

                try:
                    cam_image = model.cam(img)/255
                except:
                    cam_image=None
                
                overlayed_image, masked_image = model.lime_explain(img)
                num = 3
                
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

                model_idx += num
                print(f'Finished - {name}')
            plt.savefig(f'images/{c}.png')
            c += 1
   
        except Exception as ex:
            traceback.print_exc()
            print(f"Error {e}")
            print(ex)
            e += 1
