# Transient Object Detection in the Night Sky

This project detects transient celestial objects in astronomical images using the Ursa Minor constellation as a reference. It includes preprocessing, template matching, object comparison, and automated PDF report generation.

## Features
- Image rotation and noise reduction
- Background subtraction using Rolling Ball algorithm
- Template matching to align and crop constellation
- Object labeling and statistical comparison
- SSIM and PSNR metrics
- Automated PDF report generation

## Tech Stack
- Python
- OpenCV, NumPy, Mahotas, scikit-image, ReportLab
- Kivy and KivyMD (for UI components)

## Modules
- `info_window.py`: Handles user input and preprocessing (filtering, LAB transformation)
- `template_matcher.py`: Matches constellation using rotation + scale and crops region
- `comparator.py`: Labels and compares objects, computes metrics
- `pdf_report.py`: Generates the final project report with images and results

## How to Run
1. Install required packages:
    ```
    pip install -r requirements.txt
    ```
2. Set correct paths in each module (currently hardcoded for Windows local paths).
3. Use Kivy UI or run the backend modules independently for debugging and testing.

## Output
- Cropped and aligned images
- CSV files with region properties
- Final PDF report with visuals and object stats

## Authors
- Nisha Kazmi (CS-17040)
- Sara Jamal (CS-17052)
- Muhammad Shahzaib Ali (CS-17055)
