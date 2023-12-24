from data_preprocess import load_data
from utils import main_loop


if __name__ == "__main__":
    dset = load_data()
    main_loop(dset)
