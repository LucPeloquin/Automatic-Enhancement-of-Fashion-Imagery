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
import multiprocessing
from functools import partial

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

# Content-Aware Resizing
def content_aware_resize(image, target_width, target_height):
    """
    Resizes image while preserving important content using seam carving.
    Requires installation of: pip install pyseam
    """
    try:
        import pyseam
        img_array = np.array(image)
        resized = pyseam.seam_carve(img_array, target_width, target_height)
        return Image.fromarray(resized)
    except ImportError:
        print("PySeam not installed. Falling back to standard resize.")
        return image.resize((target_width, target_height), Image.LANCZOS)

# Automatic White Balance Correction
def auto_white_balance(image):
    """
    Applies automatic white balance correction using gray world assumption.
    """
    img_array = np.array(image).astype(np.float32)
    
    # Calculate average values for each channel
    avg_b = np.average(img_array[:, :, 0])
    avg_g = np.average(img_array[:, :, 1])
    avg_r = np.average(img_array[:, :, 2])
    avg = (avg_b + avg_g + avg_r) / 3
    
    # Calculate scaling factors
    b_factor = avg / avg_b if avg_b > 0 else 1
    g_factor = avg / avg_g if avg_g > 0 else 1
    r_factor = avg / avg_r if avg_r > 0 else 1
    
    # Apply scaling factors
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * b_factor, 0, 255)
    img_array[:, :, 1] = np.clip(img_array[:, :, 1] * g_factor, 0, 255)
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * r_factor, 0, 255)
    
    return Image.fromarray(img_array.astype(np.uint8))

# Shadow and Highlight Recovery
def recover_shadows_highlights(image, shadows=0.2, highlights=0.2):
    """
    Recovers details in shadow and highlight areas.
    
    Args:
        image: PIL Image
        shadows: Amount of shadow recovery (0-1)
        highlights: Amount of highlight recovery (0-1)
    """
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Convert to LAB
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Shadow recovery (increase brightness in dark areas)
    shadow_mask = 1.0 - l
    l_shadow = l + (shadow_mask * shadows)
    
    # Highlight recovery (decrease brightness in bright areas)
    highlight_mask = l
    l_highlight = l_shadow - (highlight_mask * highlights)
    
    # Merge channels
    enhanced_lab = cv2.merge([l_highlight, a, b])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # Convert back to 8-bit
    enhanced_rgb = np.clip(enhanced_rgb * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(enhanced_rgb)

# Texture Enhancement
def enhance_texture(image, strength=0.5):
    """
    Enhances texture details in the image.
    
    Args:
        image: PIL Image
        strength: Strength of texture enhancement (0-1)
    """
    img_array = np.array(image)
    
    # Convert to grayscale for texture detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply Laplacian filter to detect edges/texture
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    
    # Normalize Laplacian
    laplacian_norm = cv2.normalize(laplacian, None, 0, 1, cv2.NORM_MINMAX)
    
    # Create texture mask
    texture_mask = laplacian_norm * strength
    
    # Apply texture enhancement to each channel
    enhanced = img_array.copy().astype(np.float32)
    for i in range(3):
        enhanced[:,:,i] = cv2.add(
            enhanced[:,:,i], 
            gray * texture_mask * strength
        )
    
    # Clip values and convert back to uint8
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return Image.fromarray(enhanced)

# Intelligent Background Replacement
def replace_background(image, background_color=(255, 255, 255), feather_amount=3):
    """
    Removes background and replaces with solid color or pattern.
    Includes edge feathering for smoother transitions.
    
    Args:
        image: PIL Image
        background_color: RGB tuple for background color
        feather_amount: Pixels to feather at the edges
    """
    # Remove background
    no_bg = remove(image)
    no_bg_array = np.array(no_bg)
    
    # Create alpha mask
    alpha = no_bg_array[:, :, 3]
    
    # Create feathered alpha mask
    feathered_alpha = cv2.GaussianBlur(alpha, (feather_amount*2+1, feather_amount*2+1), 0)
    
    # Create new background
    bg = np.ones(no_bg_array.shape, dtype=np.uint8)
    bg[:, :, 0] = background_color[0]
    bg[:, :, 1] = background_color[1]
    bg[:, :, 2] = background_color[2]
    bg[:, :, 3] = 255
    
    # Blend foreground with background using feathered alpha
    feathered_alpha_3d = np.stack([feathered_alpha, feathered_alpha, feathered_alpha, alpha], axis=2) / 255.0
    blended = (no_bg_array * feathered_alpha_3d + bg * (1 - feathered_alpha_3d)).astype(np.uint8)
    
    return Image.fromarray(blended)

# Image Quality Assessment
def assess_image_quality(image):
    """
    Assesses image quality metrics to help determine if processing was successful.
    Returns a dictionary of quality metrics.
    """
    img_array = np.array(image)
    
    # Convert to grayscale for some metrics
    if len(img_array.shape) > 2:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Calculate sharpness using Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # Calculate contrast
    contrast = gray.std()
    
    # Calculate brightness
    brightness = gray.mean()
    
    # Calculate color diversity (if color image)
    if len(img_array.shape) > 2:
        color_std = np.std(img_array, axis=(0,1)).mean()
    else:
        color_std = 0
    
    return {
        'sharpness': sharpness,
        'contrast': contrast,
        'brightness': brightness,
        'color_diversity': color_std
    }

# Adaptive Processing Based on Image Content
def analyze_image_content(image):
    """
    Analyzes image content to determine optimal processing parameters.
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Detect if image is low light
    is_low_light = gray.mean() < 100
    
    # Detect if image is high contrast
    is_high_contrast = gray.std() > 60
    
    # Detect if image is noisy (using Laplacian)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    is_noisy = laplacian.var() > 500
    
    # Detect if image has fine details
    edges = cv2.Canny(gray, 100, 200)
    has_fine_details = np.count_nonzero(edges) > (gray.size * 0.05)
    
    return {
        'is_low_light': is_low_light,
        'is_high_contrast': is_high_contrast,
        'is_noisy': is_noisy,
        'has_fine_details': has_fine_details
    }

def get_adaptive_config(image):
    """
    Creates an adaptive configuration based on image content analysis.
    """
    analysis = analyze_image_content(image)
    
    config = {
        'upscale_factor': 4,
        'denoise_strength': 15 if analysis['is_noisy'] else 5,
        'high_pass_strength': 0.8 if analysis['has_fine_details'] else 1.2,
        'sharpen_radius': 1 if analysis['has_fine_details'] else 2,
        'sharpen_percent': 120 if analysis['is_high_contrast'] else 150,
        'sharpen_threshold': 5 if analysis['is_noisy'] else 3,
        'enhance_contrast': not analysis['is_high_contrast'],
        'enhance_colors': True,
        'saturation': 1.3 if analysis['is_low_light'] else 1.1,
        'vibrance': 1.2 if analysis['is_low_light'] else 1.0,
        'shadow_recovery': 0.3 if analysis['is_low_light'] else 0.1
    }
    
    return config

# Update the process_image function to include new enhancements
def process_image(image, config=None):
    """
    Enhanced image processing pipeline with configurable parameters and adaptive processing.
    
    Args:
        image: PIL Image to process
        config: Dictionary with processing parameters
    """
    if config is None:
        # Use adaptive configuration based on image content
        config = get_adaptive_config(image)
    
    # Start with white balance correction if enabled
    if config.get('auto_white_balance', True):
        image = auto_white_balance(image)
    
    # Apply noise reduction early
    if config.get('denoise_strength', 0) > 0:
        image = reduce_noise(image, config.get('denoise_strength', 10))
    
    # Upscale the image
    upscaled_image = upscale_image(image, config.get('upscale_factor', 4))
    
    # Enhance contrast if enabled
    if config.get('enhance_contrast', True):
        upscaled_image = enhance_contrast(upscaled_image)
    
    # Recover shadows and highlights if enabled
    if config.get('shadow_recovery', 0) > 0 or config.get('highlight_recovery', 0) > 0:
        upscaled_image = recover_shadows_highlights(
            upscaled_image,
            shadows=config.get('shadow_recovery', 0.2),
            highlights=config.get('highlight_recovery', 0.2)
        )
    
    # Apply high pass filter
    high_pass_image = high_pass_filter(upscaled_image)
    
    # Enhance texture if enabled
    if config.get('texture_enhancement', 0) > 0:
        high_pass_image = enhance_texture(
            high_pass_image,
            strength=config.get('texture_enhancement', 0.5)
        )
    
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
    
    # Replace background if specified
    if config.get('replace_background', False):
        transparent_background_image = replace_background(
            transparent_background_image,
            background_color=config.get('background_color', (255, 255, 255)),
            feather_amount=config.get('feather_amount', 3)
        )
    
    # Auto-crop based on transparent background
    autocropped_image = autocrop_image(transparent_background_image)
    
    # Assess final image quality if requested
    if config.get('assess_quality', False):
        quality_metrics = assess_image_quality(autocropped_image)
        print(f"Image quality metrics: {quality_metrics}")
    
    return autocropped_image

# Parallel Processing for Batch Operations
def parallel_batch_process(input_directory, output_directory, config=None, num_workers=None):
    """
    Process images in parallel using multiple CPU cores.
    
    Args:
        input_directory: Directory containing input images
        output_directory: Directory for output images
        config: Processing configuration
        num_workers: Number of parallel workers (defaults to CPU count)
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Get list of images
    images_with_paths = load_images(input_directory)
    if not images_with_paths:
        print("No images found, check your directory.")
        return
    
    # Create a temporary directory
    temp_directory = os.path.join(input_directory, 'temp')
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)
    
    # Define processing function for a single image
    def process_single_image(item, config):
        i, (image, path) = item
        try:
            print(f"Processing {path}")
            processed_image = process_image(image, config)
            original_filename = os.path.basename(path)
            new_input_filename = f'input_{i}{os.path.splitext(original_filename)[1]}'
            new_input_path = os.path.join(temp_directory, new_input_filename)
            shutil.move(path, new_input_path)
            output_name = f'output_{i}.png'
            save_image(processed_image, output_directory, output_name)
            return True
        except Exception as e:
            print(f"Failed to process image {path}: {e}")
            return False
    
    # Process images in parallel
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(
            partial(process_single_image, config=config),
            enumerate(images_with_paths, start=1)
        )
    
    # Move processed images back and clean up
    for file_name in os.listdir(temp_directory):
        shutil.move(os.path.join(temp_directory, file_name), input_directory)
    shutil.rmtree(temp_directory)
    
    print(f"Processed {sum(results)} images successfully.")
    print("All processing completed. Input directory cleaned up.")

# processes and renames images. inherited and used in main.py
def batch_process_images(input_directory, output_directory, config=None):
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

