from datasets import load_dataset
import random

def load_data(num_classes=30):
    dset = load_dataset(
        "imagenet-1k", split="train", streaming=True, use_auth_token=True
    )
    return dset
