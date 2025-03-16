import os
import tensorflow as tf
import pandas as pd
import cv2
import numpy as np

images_folder = "../../../images"

data = pd.DataFrame(pd.read_csv("Data_Entry_2017_v2020.csv"))

disease_data = data[data["Finding Labels"] != "No Finding"]
control_data = data[data["Finding Labels"] == "No Finding"]

diseased_pixels = []

for index, row in disease_data.iterrows():
    images_file_name = row["Image Index"]
    
    image_path = os.path.join(images_folder,images_file_name)
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        diseased_pixels.append(gray_image)
    else:
        print(f"Image file not found: {image_path}")

diseased_pixels_array = np.array(diseased_pixels)

np.save('diseased_pixels.npy', diseased_pixels_array)
print("Data Saved")

control_pixels = []

for index, row in control_data.iterrows():
    images_file_name = row["Image Index"]
    
    image_path = os.path.join(images_folder,images_file_name)
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        diseased_pixels.append(gray_image)
    else:
        print(f"Image file not found: {image_path}")

control_pixels_array = np.array(diseased_pixels)

np.save('control_pixels.npy', control_pixels_array)
print("Data Saved")
# for index , row in disease_data.iterrows(): 

#     if index >= 50:
#         break

#     images_file_name = row['Image Index']

#     image_path = os.path.join(images_folder,images_file_name)

#     if os.path.exists(image_path):
#         image = cv2.imread(image_path)
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         diseased_pixels.append(gray_image)
#     else:
#         print(f"Image file not found: {image_path}")
    

# diseased_pixels_array = np.array(diseased_pixels)

# np.save('diseased_pixels.npy', diseased_pixels_array)
# print("Data Saved")
