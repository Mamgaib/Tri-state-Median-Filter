#!/usr/bin/env python
# coding: utf-8

# # Tri-state Median Filter 

# In[19]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate  # Import tabulate library

# Function to add salt and pepper noise to an image
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size
    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

# Function to perform the Center-Weighted Median filter
def center_weighted_median(img, window_size, center_weight):
    pad_size = window_size // 2
    padded_img = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    filtered_img = np.zeros_like(img)

    for i in range(pad_size, padded_img.shape[0] - pad_size):
        for j in range(pad_size, padded_img.shape[1] - pad_size):
            kernel = padded_img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1].flatten()
            kernel = np.concatenate((kernel, np.full(center_weight - 1, padded_img[i, j])))
            filtered_img[i - pad_size, j - pad_size] = np.median(kernel)

    return filtered_img

# Modified Tri-State Median filter function
def tri_state_median_filter(img, window_size, center_weight, threshold):
    sm_filtered = cv2.medianBlur(img, window_size)
    cwm_filtered = center_weighted_median(img, window_size, center_weight)
    
    tsm_filtered = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            d1 = abs(int(img[i, j]) - int(sm_filtered[i, j]))
            d2 = abs(int(img[i, j]) - int(cwm_filtered[i, j]))

            if d2 < threshold:
                if d1 < threshold:
                    tsm_filtered[i, j] = img[i, j]  # Keep original
                else:
                    tsm_filtered[i, j] = sm_filtered[i, j]  # Use SM
            else:
                tsm_filtered[i, j] = cwm_filtered[i, j]  # Use CWM

    return tsm_filtered

# Function to perform a standard Median filter from scratch
def custom_median_filter(img, window_size):
    pad_size = window_size // 2
    filtered_img = np.zeros_like(img)

    for i in range(pad_size, img.shape[0] - pad_size):
        for j in range(pad_size, img.shape[1] - pad_size):
            kernel = img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            median_value = np.median(kernel)
            filtered_img[i, j] = median_value

    return filtered_img

# Function to calculate MSE
def calculate_mse(original, filtered):
    return np.mean((original - filtered) ** 2)

# Function to compute PSNR
def compute_psnr(original, filtered):
    mse = calculate_mse(original, filtered)
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

# Load the image using cv2
image = cv2.imread(r'C:\Users\User\Desktop\veggies.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image loaded successfully
if image is None:
    print("Error: Unable to load the image.")
    exit()

# Add salt and pepper noise to the image
noised_image = add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02)

# Apply custom standard median filter
median_filtered = custom_median_filter(noised_image, window_size=3)

# Apply Center-Weighted Median filter
cwm_filtered = center_weighted_median(noised_image, window_size=3, center_weight=3)

# Apply Tri-State Median filter
tsm_filtered = tri_state_median_filter(noised_image, window_size=3, center_weight=3, threshold=25)

# Calculate MSE and PSNR for each filtered output
mse_sm = calculate_mse(image, median_filtered)
mse_cwm = calculate_mse(image, cwm_filtered)
mse_tsm = calculate_mse(image, tsm_filtered)

psnr_sm = compute_psnr(image, median_filtered)
psnr_cwm = compute_psnr(image, cwm_filtered)
psnr_tsm = compute_psnr(image, tsm_filtered)

# Create a table for MSE and PSNR values
table_data = [
    ["Filter", "MSE", "PSNR"],
    ["Standard Median Filter", f"{mse_sm:.2f}", f"{psnr_sm:.2f}"],
    ["Center-Weighted Median Filter", f"{mse_cwm:.2f}", f"{psnr_cwm:.2f}"],
    ["Tri-State Median Filter", f"{mse_tsm:.2f}", f"{psnr_tsm:.2f}"],
]

# Print the table
print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))

# Display the images
plt.figure(figsize=(16, 12))

plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(2, 3, 2)
plt.title('Noised Image')
plt.imshow(noised_image, cmap='gray')

plt.subplot(2, 3, 3)
plt.title(f'Standard Median Filter (PSNR={psnr_sm:.2f}, MSE={mse_sm:.2f})')
plt.imshow(median_filtered, cmap='gray')

plt.subplot(2, 3, 4)
plt.title(f'Center-Weighted Median Filter (PSNR={psnr_cwm:.2f}, MSE={mse_cwm:.2f})')
plt.imshow(cwm_filtered, cmap='gray')

plt.subplot(2, 3, 5)
plt.title(f'Tri-State Median Filter (PSNR={psnr_tsm:.2f}, MSE={mse_tsm:.2f})')
plt.imshow(tsm_filtered, cmap='gray')

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




