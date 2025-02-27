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
    # Increase kernel size for more control over high-pass effect
    low_pass_L = cv2.GaussianBlur(L, (5, 5), 0)
    high_pass_L = cv2.subtract(L, low_pass_L)
    # Add strength parameter to control high-pass intensity
    strength = 1.2  # Adjustable parameter (1.0 = original strength)
    high_pass_L = cv2.multiply(high_pass_L, strength)
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
    
# Add a new function for color enhancement
def enhance_colors(image, saturation_factor=1.2, vibrance_factor=1.1):
    """
    Enhances colors by adjusting saturation and vibrance.
    Vibrance increases saturation of less-saturated colors more than already-saturated ones.
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to HSV for better color manipulation
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Saturation adjustment (uniform)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
    
    # Vibrance (selective saturation)
    # Calculate current saturation level
    sat_mask = hsv[:, :, 1] / 255.0
    # Apply more saturation to less saturated pixels
    vibrance_effect = (1 - sat_mask) * (vibrance_factor - 1)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + vibrance_effect), 0, 255)
    
    # Convert back to RGB
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(enhanced)

# Add noise reduction function
def reduce_noise(image, strength=10):
    """
    Applies non-local means denoising to reduce noise while preserving details.
    """
    img_array = np.array(image)
    # Apply non-local means denoising
    denoised = cv2.fastNlMeansDenoisingColored(
        img_array, 
        None, 
        strength,  # Filter strength
        strength,  # Color component filter strength
        7,         # Template window size
        21         # Search window size
    )
    return Image.fromarray(denoised)

# Add contrast enhancement function
def enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Enhances contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    which works better than global contrast adjustment.
    """
    img_array = np.array(image)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    # Merge channels
    enhanced_lab = cv2.merge((cl, a, b))
    
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(enhanced_rgb)

# Update the process_image function to include new enhancements
def process_image(image, config=None):
    """
    Enhanced image processing pipeline with configurable parameters.
    
    Args:
        image: PIL Image to process
        config: Dictionary with processing parameters
    """
    if config is None:
        config = {
            'upscale_factor': 4,
            'denoise_strength': 10,
            'high_pass_strength': 1.2,
            'sharpen_radius': 2,
            'sharpen_percent': 150,
            'sharpen_threshold': 3,
            'enhance_contrast': True,
            'enhance_colors': True,
            'saturation': 1.2,
            'vibrance': 1.1
        }
    
    # Start with noise reduction (better to do this early)
    if config.get('denoise_strength', 0) > 0:
        image = reduce_noise(image, config.get('denoise_strength', 10))
    
    # Upscale
    upscaled_image = upscale_image(image, config.get('upscale_factor', 4))
    
    # Enhance contrast if enabled
    if config.get('enhance_contrast', True):
        upscaled_image = enhance_contrast(upscaled_image)
    
    # Apply high pass filter
    high_pass_image = high_pass_filter(upscaled_image)
    
    # Apply unsharp mask
    sharpened_image = unsharp_mask(
        high_pass_image, 
        radius=config.get('sharpen_radius', 2),
        percent=config.get('sharpen_percent', 150),
        threshold=config.get('sharpen_threshold', 3)
    )
    
    # Enhance colors if enabled
    if config.get('enhance_colors', True):
        sharpened_image = enhance_colors(
            sharpened_image,
            saturation_factor=config.get('saturation', 1.2),
            vibrance_factor=config.get('vibrance', 1.1)
        )
    
    # Remove background
    transparent_background_image = remove(sharpened_image)
    
    # Auto-crop based on transparent background
    autocropped_image = autocrop_image(transparent_background_image)
    
    return autocropped_image

# processes and renames images. inherited and used in main.py
def batch_process_images(input_directory, output_directory, config=None):
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
            processed_image = process_image(image, config)
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

