import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def process_Unet_pred(folder_path, predictions):
  
    predictions = np.squeeze(predictions, axis=0)
    rescaled_predictions = (predictions * 255).astype(np.uint8)
    min_val = np.min(rescaled_predictions)
    max_val = np.max(rescaled_predictions)
    stretched_predictions = ((rescaled_predictions - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    flattened_predictions = stretched_predictions.flatten()

    # Create a list to hold the images
    images = []

    # List all files in the directory specified by folder_path
    for filename in os.listdir(folder_path):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)

        # Read the image file
        img = cv2.imread(file_path)

        # Check if the image file was successfully read
        if img is not None:
            # Append the image to the list
            images.append(img)

    # Check if any images were found
    if len(images) == 0:
        print(f"No images found in folder: {folder_path}")
        return None

    # Convert the list of images to a numpy array
    images_np = np.array(images)

    # Create the exposure fusion object
    merge_mertens = cv2.createMergeMertens()

    # Fuse the images
    fusion = merge_mertens.process(images_np)

    # The fusion image is in the range of [0, 1], so it needs to be scaled to the range [0, 255] for a standard image
    fusion_scaled = np.clip(fusion*255, 0, 255).astype(np.uint8)

    # Return the fused image
    return flattened_predictions, fusion_scaled
