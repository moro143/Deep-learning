from data_preprocess import load_data
from utils import check_num_classes, predict_images


if __name__ == "__main__":
    dset = load_data()
    predict_images(dset)
