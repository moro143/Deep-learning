from datasets import load_dataset
import random


def load_data(num_classes=30):
    dset = load_dataset(
        "imagenet-1k", split="train", streaming=True, use_auth_token=True
    )

    class_labels = set()
    for sample in dset:
        class_labels.add(sample['label'])
        if len(class_labels) >= num_classes:
            break

    selected_classes = random.sample(list(class_labels), num_classes)

    filtered_dset = dset.filter(lambda x: x['label'] in selected_classes)

    return filtered_dset
