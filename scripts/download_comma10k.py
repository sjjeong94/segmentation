from segmentation.utils.helpers import gdown_and_extract


if __name__ == "__main__":
    dataset_id = "1N8u6XyY6ddYt2eLiY__Nskh5aZ7Nxrk2"
    dataset_dir = "dataset/"
    gdown_and_extract(dataset_id, dataset_dir)
