# Automatic Enhancement of Fashion Imagery

This application provides an automated solution for enhancing fashion product images, particularly those from platforms like Grailed.com. It addresses common issues in user-submitted fashion photography such as poor lighting, low resolution, distracting backgrounds, and lack of detail clarity.

## Running the Application

You can run this program by executing the ui.py file:
```
python ui.py
```

## Features

- Adaptive image processing based on content analysis
- High-quality 4x upscaling using Lanczos algorithm
- Detail enhancement through high-pass filtering and unsharp masking
- Color enhancement with LAB color space processing
- Contrast improvement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Shadow and highlight recovery
- Noise reduction using non-local means denoising
- Background removal using the U2Net deep learning model
- Auto-cropping based on content
- Batch processing with parallel execution capabilities
- User-friendly graphical interface

## Requirements

- Python 3.12
- Required libraries:
  - opencv-python: For advanced image processing operations
  - numpy: For numerical operations and array handling
  - Pillow: For basic image manipulation and format conversion
  - rembg: For background removal using the U2Net deep learning model
  - plyer: For desktop notifications in the UI

- Optional dependencies:
  - pyseam: For content-aware resizing functionality

See requirements.txt for specific version requirements.

## Input/Output

- This application accepts .jpg, .jpeg, and .png files as valid inputs
- Output images are saved as PNG to preserve transparency
- Input images are preserved (not deleted) during processing

## File Structure

- ui.py: The interface file with the graphical user interface
- main.py: The core algorithm file containing all image processing functions
- requirements.txt: List of required Python libraries
- final_report_peloqj1.docx / final_report_peloqj1.pdf: Detailed project report
- final_presentation_peloqj1.pptx / final_presentation_peloqj1.pdf: Project presentation

## Standalone Version

There is a standalone executable version of the program in dist/ui/ui.exe that has been compiled for Windows. This version has not been tested on any system other than Windows.

## Sample Images

Sample images are provided in the "input" and "in" folders. The "output" and "out" folders contain the corresponding processed images.

Note: There is no input folder uploaded in the repository due to copyright concerns.

## Processing Time

Processing time depends on image size and resolution, but typically takes around 6 seconds per high-resolution image on an average computer, significantly faster than manual enhancement through applications like Photoshop.
