import tensorflow as tf
import pandas as pd
import cv2

prune_train = pd.read_csv("PruneCXR/miccai2023_nih-cxr-lt_labels_train.csv")
print(prune_train)

prune_test = pd.read_csv("PruneCXR/miccai2023_nih-cxr-lt_labels_test.csv")

prune_extra = pd.read_csv("PruneCXR/miccai2023_nih-cxr-lt_labels_val.csv")

#filter the data to only include the rows with findings
prune_train = prune_train[prune_train["No Finding"] == 0]
prune_test = prune_test[prune_test["No Finding"] == 0]
prune_extra = prune_extra[prune_extra["No Finding"] == 0]

#loop through the csvs by the image name to convert image data to pixels
test = "../../images00000001_000.png"
img = cv2.imread(test,cv2.IMREAD_GRAYSCALE)
#for iter in range(len(prune_train)):
 #   #print(prune_train[iter])
  #  print("___________")
















