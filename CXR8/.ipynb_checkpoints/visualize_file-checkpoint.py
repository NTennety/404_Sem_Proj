import numpy as np

# Load the NumPy file
data = np.load('test_images_pixels.npy')

# Print the shape of the array
print(f'Shape of the array: {data.shape}')

# Print the first image pixel array
print('First image pixel array:')
print(data[0])
