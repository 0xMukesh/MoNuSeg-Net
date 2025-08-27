import os
from PIL import Image
import numpy as np

DATASET_ROOT = "./data/train/masks"


def main():
    patients = os.listdir(DATASET_ROOT)
    cell_type_cnts = [0, 0, 0, 0]
    total_imgs = 0

    for patient in patients:
        patient_dir_path = os.path.join(DATASET_ROOT, patient)
        img_dirs = os.listdir(patient_dir_path)

        for img_dir in img_dirs:
            img_files = os.listdir(os.path.join(patient_dir_path, img_dir))

            for img_file in img_files:
                if img_file == "in.png":
                    total_imgs += 1
                    continue

                class_idx = int(img_file.split(".png")[0])

                if class_idx not in range(0, 4):
                    continue

                img = Image.open(os.path.join(patient_dir_path, img_dir, img_file))
                img_array = np.array(img)

                if not np.all(img_array == 0):
                    cell_type_cnts[class_idx] += 1

    print(f"no. of cells by type - {cell_type_cnts}")
    print(f"total no. of imgs - {total_imgs}")


if __name__ == "__main__":
    main()
