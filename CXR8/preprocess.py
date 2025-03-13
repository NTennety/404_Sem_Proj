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



image_pixels = []

# Loop through each row in the DataFrame
for index, row in prune.iterrows():
    # Extract the image file name
    image_file_name = row['id']
    
    # Create the full path to the image file
    image_path = os.path.join(images_folder, image_file_name)
    
    # Check if the image file exists
    if os.path.exists(image_path):
        # Read the image using cv2
        image = cv2.imread(image_path)
        
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get the pixel values and add to the list
        image_pixels.append(gray_image)
    else:
        print(f"Image file not found: {image_path}")

# Convert the list to a NumPy array
image_pixels_array = np.array(image_pixels)

# Save the NumPy array to a file
np.save('test_images_pixels.npy', image_pixels_array)

print("Image pixel data has been saved to test_images_pixels.npy")














