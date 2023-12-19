from datasets import load_dataset


def load_data():
    dset = load_dataset(
        "imagenet-1k", split="train", streaming=True, use_auth_token=True
    )
    return dset
