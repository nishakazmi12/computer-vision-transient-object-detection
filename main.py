# main.py - End-to-end runner for Transient Object Detection project

from info_window import InfoWindow
from template_matcher import TemplateMatcher
from comparator import Comparator
from pdf_report import PDFReport
import cv2

## D:\FinalProject\transient_object_detection_modular

# Define your image paths here
source_img = "transient_object_detection_modular/Transient_Images"
out_source = "transient_object_detection_modular/out"
staging_img = "transient_object_detection_modular/staging"


merge_crop_img_path = out_source + "/corrected_merge.TIF"
background_source_img_path = out_source + "/img_without_background_source.TIF"
logo_path = "transient_object_detection_modular/logo.jpg"
pdf_path = "transient_object_detection_modular/PDFReport.pdf"


# Step 1: Preprocess input image
print("Running preprocessing...")
info = InfoWindow()
info.input_vals = {"d": 5, "gs": 50, "gr": 50, "radius": 15.0}  # manually input values
img1 = info.apply_bilateral_filter(source_img + "/UPLOADED.TIF")
scaled_img1 = info.resize_and_save(img1, "/bil_scaled.TIF")
merged, img_wo_bg, bg = info.process_lab(scaled_img1)
# prefix
info.save_processed_images(out_source, merged, img_wo_bg, bg)

# Step 2: Template Matching and Cropping
print("Running template matching...")
# image_gray, source_dir, merge_crop_img
template_matcher = TemplateMatcher(cv2.imread(out_source + "/img_without_background.TIF", 0),
                                   source_img,
                                   cv2.imread(out_source + "/corrected_merge.TIF"))
template_matcher.find_best_match()
template_matcher.crop_match_area()

# Step 3: Comparison
print("Running object comparison...")
comparator = Comparator(staging_img + "/cropped.TIF",
                        background_source_img_path) # or maybe source_img + "/UPLOADED.TIF"
comparator.compare_images()

# Step 4: Generate Report
print("Generating PDF report...")
report = PDFReport(pdf_path)
report.add_title_page(logo_path, "Transient Object Detection", [
    ["Nisha Kazmi", "CS-17040"],
    ["Sara Jamal", "CS-17052"]
])
report.add_paragraph("This project detects transient celestial objects in night sky images using image analysis and comparison techniques.")
report.add_image("transient_object_detection_modular/DetectedObjects.TIF", "Detected Transient Objects")
report.add_image("transient_object_detection_modular/DetectedObjects_Source.TIF", "Reference Image with Detection")
report.build_report()

print("✅ Project pipeline completed.")
