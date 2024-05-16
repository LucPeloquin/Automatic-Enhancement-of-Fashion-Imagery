# Image Enhancement code
# methodology for automatically enhancing and abackground subtraction of an image
# which is the backbone of the ui.py file. Images trained were fashion images from Grailed.com listings
#
# this script alters the image by using Lancoz (4x) for upscaline, high-pass filtering, unsharp masking,
# auto-cropping, and rembg for background removal. Input is one or many files and is automatically renamed for
# research and refinement purpouses. Default output folder is output in the same directory as the input folder.
#
# note: no input images are deleted if this program is stopped before completion, just moved to input/temp for renaming purpouses
#
# created by seven / Jean-Luc Peloquin

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageChops
from rembg import remove
import os
import shutil

# upscales the image using Lancoz
def upscale_image(image, scale_factor=4):
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    upscaled_image = image.resize((new_width, new_height), Image.LANCZOS)
    return upscaled_image

# applying highpass filter
def high_pass_filter(image):
    image_lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(image_lab)
    low_pass_L = cv2.GaussianBlur(L, (3, 3), 0)
    high_pass_L = cv2.subtract(L, low_pass_L)
    L_high_pass = cv2.add(L, high_pass_L)
    processed_lab = cv2.merge([L_high_pass, A, B])
    processed_rgb = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(processed_rgb)

# apply unsharp mask while maintaining color integrty
def unsharp_mask(image, radius=2, percent=150, threshold=3):
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    mask = ImageChops.subtract(image, blurred)
    enhanced_mask = mask.point(lambda x: x * (percent / 100))
    if threshold > 0:
        thresholded_mask = enhanced_mask.point(lambda x: 0 if x < threshold else x)
    else:
        thresholded_mask = enhanced_mask
    sharpened_image = ImageChops.add(image, thresholded_mask)
    return sharpened_image

# another implementation
# return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

# loads imagesxs from specified directory
def load_images(directory):
    images = []
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return images
    for filename in sorted(os.listdir(directory), key=str.casefold):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(directory, filename)
            try:
                img = Image.open(img_path).convert('RGBA')
                images.append((img, img_path))
            except IOError:
                print(f"Error opening {img_path}")
    return images

# saves image
def save_image(image, path, file_name):
    if not os.path.exists(path):
        os.makedirs(path)
    image.save(os.path.join(path, file_name))

# automatically crops the image
def autocrop_image(image_pil):
    image_cv = np.array(image_pil)
    if image_cv.shape[2] == 4:
        _, alpha = cv2.threshold(image_cv[:, :, 3], 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Crop the original image to the bounding box
            cropped_image_cv = image_cv[y:y+h, x:x+w]
            cropped_image_pil = Image.fromarray(cropped_image_cv)
            return cropped_image_pil
    # if no transparancy detected
    return image_pil
    
# pipeline
def process_image(image):
    upscaled_image = upscale_image(image)   # Upscales 4x
    high_pass_image = high_pass_filter(upscaled_image)  # Applies a high pass filter
    sharpened_image = unsharp_mask(high_pass_image)  # Applies unsharp mask
    transparent_background_image = remove(sharpened_image)    # Removes background
    autocropped_image = autocrop_image(transparent_background_image)    # Auto-crops based on transparent background
    return autocropped_image

# processes and renames images. inherited and used in main.py
def batch_process_images(input_directory, output_directory):
    pass
    print(f"Loading images from {input_directory}")
    images_with_paths = load_images(input_directory)
    if not images_with_paths:
        print("No images found, check your directory.")
        return
    
    # Temporary directory for processing
    temp_directory = os.path.join(input_directory, 'temp')
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    print(f"Processing images...")
    for i, (image, path) in enumerate(images_with_paths, start=1):
        try:
            print(f"Processing {path}")
            processed_image = process_image(image)
            original_filename = os.path.basename(path)
            new_input_filename = f'input_{i}{os.path.splitext(original_filename)[1]}'
            new_input_path = os.path.join(temp_directory, new_input_filename)
            shutil.move(path, new_input_path)
            output_name = f'output_{i}.png'
            save_image(processed_image, output_directory, output_name)
            print(f"Saved processed image to {output_directory}/{output_name}")
        except Exception as e:
            print(f"Failed to process image {path}: {e}")

    # Move processed images back to original directory and clean up
    for file_name in os.listdir(temp_directory):
        shutil.move(os.path.join(temp_directory, file_name), input_directory)
    shutil.rmtree(temp_directory)

    print("All processing completed. Input directory cleaned up.")

