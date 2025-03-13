import os
import tensorflow as tf
import pandas as pd
import cv2
import numpy as np

train_csv_path = "PruneCXR/miccai2023_nih-cxr-lt_labels_train.csv"

test_csv_path = "PruneCXR/miccai2023_nih-cxr-lt_labels_test.csv"

prune_extra_path = "PruneCXR/miccai2023_nih-cxr-lt_labels_val.csv"

images_folder = "../../final_images"

prune_train = pd.read_csv(train_csv_path)
prune_test = pd.read_csv(test_csv_path)
prune_extra = pd.read_csv(prune_extra_path)

prune_train = prune_train[prune_train["No Finding"] == 0]
prune_test = prune_test[prune_test["No Finding"] == 0]
prune_extra = prune_extra[prune_extra["No Finding"] == 0]

train_pixels = []

for index, row in prune_train.iterrows():
    if index >= 50:
        break
    
    image_file_name = row['id']
    image_path = os.path.join(images_folder, image_file_name)
    
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        train_pixels.append(gray_image)
    else:
        print(f"Image file not found: {image_path}")

train_pixels_array = np.array(train_pixels)
np.save('train_images_pixels.npy', train_pixels_array)
print("Image pixel data has been saved to train_images_pixels.npy")

test_pixels = []

for index, row in prune_test.iterrows():
    if index >= 50:
        break
    
    image_file_name = row['id']
    image_path = os.path.join(images_folder, image_file_name)
    
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_pixels.append(gray_image)
    else:
        print(f"Image file not found: {image_path}")

test_pixels_array = np.array(test_pixels)
np.save('test_images_pixels.npy', test_pixels_array)
print("Image pixel data has been saved to test_images_pixels.npy")

extra_pixels = []

for index, row in prune_extra.iterrows():
    if index >= 50:
        break
    
    image_file_name = row['id']
    image_path = os.path.join(images_folder, image_file_name)
    
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        extra_pixels.append(gray_image)
    else:
        print(f"Image file not found: {image_path}")

extra_pixels_array = np.array(extra_pixels)
np.save('extra_images_pixels.npy', extra_pixels_array)
print("Image pixel data has been saved to extra_images_pixels.npy")

