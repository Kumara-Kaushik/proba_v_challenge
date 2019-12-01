import os
import shutil
import numpy as np
from glob import glob
import cv2 as cv2
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from zipfile import ZipFile
import fire
import tqdm
import wget


def prepare_dataset(root_path="./probav_data", download=False):

    if not os.path.exists(os.path.join(root_path, "probav_data")):
        os.mkdir(os.path.join(root_path, "probav_data"))

    if download:
        print("Downloading Dataset...")
        url = "https://kelvins.esa.int/media/competitions/proba-v-super-resolution/probav_data.zip"
        wget.download(url, './probav_data/probav_data.zip')
        print("Extracting the data...")
        ZipFile(os.path.join(root,'probav_data.zip')).extractall(root+'/')
        print("dataset downloaded and extracted!")
        os.remove(os.path.join(root,'probav_data.zip'))

        if os.path.exists(os.path.join(root, "readme.txt")):
            os.remove(os.path.join(root, "readme.txt"))

    train_path = os.path.join(root, "train")
    test_path = os.path.join(root, "test")

    print("Preparing training data...")
    save_images(train_path)
    print("Training data preparation done!")
    print("Preparing testing data...")
    save_images(test_path)
    print("Testing data preparation done!")

def save_images(db_path):
    hr_path = os.path.join(db_path, "HR_imgs")
    lr_path = os.path.join(db_path, "LR_imgs")

    if not os.path.exists(hr_path):
        os.mkdir(hr_path)
    if not os.path.exists(lr_path):
        os.mkdir(lr_path)

    path_walk = next(os.walk(db_path))

    for path in [os.path.join(path_walk[0], i) for i in path_walk[1] if i in ["NIR", "RED"]]:
        for img_folder_path in glob(path + "/" + "imgset*"):
            scene_path = img_folder_path + "/"

            hr_img_path = os.path.join(scene_path + "HR.png")
            if os.path.exists(hr_img_path):
                hr_img = cv2.imread(hr_img_path, 0)
                hr_save_path = hr_path + "/" + img_folder_path[-10:] + ".png"
                if not os.path.exists(hr_save_path):
                    cv2.imwrite(hr_save_path, hr_img)
                # shutil.copy(scene_path + "HR.png", hr_path + "/" + img_folder_path[-10:] + ".png")

            images = []
            for lr_img_path in glob(scene_path + 'LR*.png'):
                lr_mask_path = lr_img_path.replace('LR', 'QM')
                lr_img = cv2.imread(lr_img_path, 0)
                lr_mask = cv2.imread(lr_mask_path, 0)
                images.append((lr_img,lr_mask))

            modified_lr_img = return_modified_lr_img(images)
            lr_save_path = lr_path + "/" + img_folder_path[-10:] + ".png"
            if not os.path.exists(lr_save_path):
                cv2.imwrite(lr_save_path, modified_lr_img)
            print(f"Folder {img_folder_path} Done!")

def return_modified_lr_img(images, average=True):

    (h, w)= images[0][0].shape
    c = len(images)
    img_array = np.zeros((h, w, c))
    mask_array = np.zeros((h, w, c))
    final_lr_img = np.zeros((h, w))

    for i in range(c):
        img_array[:, :, i] = images[i][0]
        mask_array[:, :, i] = images[i][1]

    best_channel = return_best_mask_channel(mask_array)
    final_lr_img = img_array[:, :, best_channel]

    for i in range(h):
        for j in range(w):
            if mask_array[i, j, best_channel] == 0.0:
                mask_c_list = mask_array[i, j, :].tolist()
                img_c_list = img_array[i, j, :].tolist()
                if sum(mask_c_list) != 0:
                    unobstructed_weight_list = [img_c_list[i] for i in range(len(img_c_list)) if mask_c_list[i] != 0]
                    avg_weight = np.average(unobstructed_weight_list)
                    median_weight = np.median(unobstructed_weight_list)
                else:
                    avg_weight = np.average(img_c_list)
                    median_weight = np.median(img_c_list)
                if average:
                    final_lr_img[i, j] = int(avg_weight)
                else:
                    final_lr_img[i, j] = int(median_weight)
    return final_lr_img


def return_best_mask_channel(mask_array):
    sum_dict = {i: mask_array[:, :, i].sum() for i in range(len(mask_array[0, 0, :]))}
    best_mask_c = sorted(sum_dict.items(), key=lambda k: k[1])[-1][0]
    return best_mask_c


if __name__ == "__main__":
    fire.Fire(prepare_dataset)