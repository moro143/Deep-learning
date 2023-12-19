from models import predict_vgg16, predict_resnet, predict_efficientnet


def check_num_classes(dataset, target_num_classes=30):
    class_labels = set()
    for sample in dataset:
        class_labels.add(sample["label"])
        if len(class_labels) == target_num_classes:
            return True

    return len(class_labels) == target_num_classes


def predict_images(dset):
    e = 0
    for i in dset:
        try:
            width = 35
            print(f"VGG Prediction:".ljust(width) + str(predict_vgg16(i["image"])[0]))
            print(f"ResNet Prediction:".ljust(width) + str(predict_resnet(i["image"])[0]))
            print(
                f"EfficientNet Prediction:".ljust(width)
                + str(predict_efficientnet(i["image"])[0])
            )

        except Exception as ex:
            print(f"Error {e}")
            print(ex)
            e += 1
