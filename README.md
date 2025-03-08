# Automatic Enhancement of Fashion Imagery

This application provides an automated solution for enhancing fashion product images, particularly those from platforms like Grailed.com. It addresses common issues in user-submitted fashion photography such as poor lighting, low resolution, distracting backgrounds, and lack of detail clarity.

## Overview

The project implements a comprehensive image enhancement pipeline with adaptive processing capabilities. The system analyzes each image to determine optimal processing parameters based on lighting conditions, contrast levels, noise presence, and detail complexity, ensuring tailored enhancements for different types of fashion images.

## Features

- **Adaptive image processing** based on content analysis
- **High-quality upscaling** using Lanczos algorithm (4x by default)
- **Detail enhancement** through high-pass filtering and unsharp masking
- **Color enhancement** with LAB color space processing
- **Contrast improvement** using CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Shadow and highlight recovery** for better dynamic range
- **Noise reduction** using non-local means denoising
- **Background removal** using the U2Net deep learning model
- **Auto-cropping** based on content
- **Batch processing** with parallel execution capabilities
- **User-friendly graphical interface**

## Enhancement Pipeline

The enhancement pipeline consists of the following stages:

1. **Image Analysis**: The system analyzes each image to determine optimal parameters
2. **Pre-processing**: White balance correction and noise reduction
3. **Resolution Enhancement**: Lanczos upscaling
4. **Color and Contrast Enhancement**: CLAHE, shadow/highlight recovery, color enhancement
5. **Detail Enhancement**: High-pass filtering, texture enhancement, unsharp masking
6. **Background Processing**: Background removal, optional replacement, edge feathering
7. **Post-processing**: Auto-cropping and quality assessment

## Requirements

- Python 3.12 or higher
- Required libraries:
  - opencv-python: For advanced image processing operations
  - numpy: For numerical operations and array handling
  - Pillow: For basic image manipulation and format conversion
  - rembg: For background removal using the U2Net deep learning model
  - plyer: For desktop notifications in the UI

- Optional dependencies:
  - pyseam: For content-aware resizing functionality

See requirements.txt for specific version requirements.

## Installation

```bash
# Clone the repository
git clone https://github.com/LucPeloquin/Automatic-Enhancement-of-Fashion-Imagery.git
cd Automatic-Enhancement-of-Fashion-Imagery

# Install required dependencies
pip install -r requirements.txt
```

## Usage

### Using the Graphical Interface

1. Run the UI application:
   ```bash
   python ui.py
   ```
2. Use the interface to:
   - Select an input folder containing images
   - Choose an output folder (or use the default)
   - Start the processing

### Using the Command Line (Advanced Users)

For direct access to the enhancement functions, you can import and use the main.py module in your Python scripts:

```python
import main

# Process a single image
from PIL import Image
image = Image.open('path/to/image.jpg')
enhanced_image = main.process_image(image)
enhanced_image.save('path/to/output.png')

# Process a folder of images
main.batch_process_images('path/to/input_folder', 'path/to/output_folder')

# Process images in parallel
main.parallel_batch_process('path/to/input_folder', 'path/to/output_folder')
```

## Customizing Enhancement Parameters

Advanced users can modify the enhancement parameters:

```python
# Example of custom configuration
custom_config = {
    'upscale_factor': 2,  # Lower for faster processing
    'denoise_strength': 10,
    'high_pass_strength': 1.0,
    'sharpen_radius': 2,
    'sharpen_percent': 130,
    'sharpen_threshold': 3,
    'enhance_contrast': True,
    'enhance_colors': True,
    'saturation': 1.1,
    'vibrance': 1.0,
    'shadow_recovery': 0.2
}

# Use custom config with processing functions
main.process_image(image, custom_config)
main.batch_process_images(input_dir, output_dir, custom_config)
```

## Input/Output

- This application accepts .jpg, .jpeg, and .png files as valid inputs
- Output images are saved as PNG to preserve transparency
- Input images are preserved (not deleted) during processing

## File Structure

- `ui.py`: The interface file with the graphical user interface
- `main.py`: The core algorithm file containing all image processing functions
- `requirements.txt`: List of required Python libraries
- `final_report_peloqj1.docx` / `final_report_peloqj1.pdf`: Detailed project report
- `final_presentation_peloqj1.pptx` / `final_presentation_peloqj1.pdf`: Project presentation

## Technical Implementation

### Key Algorithms

#### Lanczos Upscaling
The Lanczos algorithm uses a windowed sinc function for resampling, providing superior quality for upscaling fashion images where detail preservation is critical.

#### High-Pass Filtering
High-pass filtering enhances fine details by removing low-frequency components and emphasizing high-frequency ones. The implementation uses LAB color space to preserve color accuracy while enhancing details.

#### Unsharp Masking
Unsharp masking enhances edges by subtracting a blurred version of the image from the original, then adding the difference back with amplification.

#### Adaptive Contrast Enhancement
CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances contrast locally, which is particularly effective for fashion images with varying lighting conditions.

#### Background Removal
The project uses the rembg library, which implements the U2Net deep learning model for background removal. This provides high-quality segmentation of fashion items from their backgrounds.

### Adaptive Configuration System

The adaptive configuration system analyzes each image to determine optimal processing parameters:

```python
def analyze_image_content(image):
    # Detect if image is low light
    is_low_light = gray.mean() < 100
    
    # Detect if image is high contrast
    is_high_contrast = gray.std() > 60
    
    # Detect if image is noisy
    is_noisy = laplacian.var() > 500
    
    # Detect if image has fine details
    has_fine_details = np.count_nonzero(edges) > (gray.size * 0.05)
```

Based on this analysis, the system creates a tailored configuration for each image.

## Performance Considerations

- Memory usage is optimized by processing images sequentially in batch mode
- Parallel processing is available for systems with multiple cores
- Temporary files are managed to prevent disk space issues
- Processing parameters can be adjusted for performance vs. quality tradeoffs
- Processing time is approximately 6 seconds per high-resolution image on an average computer

## Standalone Version

There is a standalone executable version of the program in dist/ui/ui.exe that has been compiled for Windows. This version has not been tested on any system other than Windows.

## Sample Images

Sample images are provided in the "input" and "in" folders. The "output" and "out" folders contain the corresponding processed images.

Note: There is no input folder uploaded in the repository due to copyright concerns.

## Limitations

- Background removal quality depends on contrast between subject and background
- Very low-resolution images may not provide enough detail for successful enhancement
- Processing very large images may require more memory

## Future Development

Potential improvements include:
- AI-based object recognition for smarter cropping
- Style transfer options for consistent product presentation
- Batch configuration profiles for different product categories
- Cloud processing integration for higher throughput
- Mobile application version
