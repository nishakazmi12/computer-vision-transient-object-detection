# Unified Transient Object Detection Script (Optimized)
# Contains: InfoWindow, TemplateMatcher, Comparator, PDFReport

import os
import cv2
import imutils
import glob
import numpy as np
import pandas as pd
import mahotas as mh # type: ignore
from threading import Thread
from matplotlib import pyplot as plt
from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock, mainthread
from cv2_rolling_ball import subtract_background_rolling_ball as sbrb # type: ignore
from skimage import measure
from skimage.segmentation import clear_border
from skimage.metrics import structural_similarity as compare_ssim
from reportlab.lib.pagesizes import A3
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Paths
path_main = "transient_object_detection_modular"
pathout = path_main + "/Transient_Images/"
path_for_source = path_main + "/out/"

class InfoWindow(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_vals = {"d": 5, "gs": 0, "gr": 0, "radius": 13.0}

    def process_inputs(self):
        try:
            self.input_vals["gs"] = int(self.ids.input1.text)
            self.input_vals["gr"] = int(self.ids.input2.text)
            radius = float(self.ids.input3.text)
            self.input_vals["radius"] = radius if 12 <= radius <= 30 else 13.0
        except ValueError:
            self.input_vals["gs"] = self.input_vals["gr"] = 50
            self.input_vals["radius"] = 13.0

        with open(path_main + "UserInput.txt", "w") as f:
            f.write(f"Spacial Parameter : {self.input_vals['gs']} \nRange Parameter: {self.input_vals['gr']} \nRadius: {self.input_vals['radius']}")

    def apply_bilateral_filter(self, image_path):
        img = cv2.imread(image_path)
        return cv2.bilateralFilter(img, self.input_vals["d"], self.input_vals["gs"], self.input_vals["gr"], borderType=cv2.BORDER_CONSTANT)

    def resize_and_save(self, img, filename):
        resized = imutils.resize(img, width=int(img.shape[1] * 0.6))
        cv2.imwrite(pathout + filename, resized)
        return resized

    def process_lab(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        img_wo_bg, bg = sbrb(l, self.input_vals["radius"], light_background=False, use_paraboloid=False, do_presmooth=False)
        merged = cv2.merge((img_wo_bg, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR), img_wo_bg, bg

    def save_processed_images(self, prefix, img, wo_bg, bg):
        self.resize_and_save(img, f"{prefix}corrected_merge.TIF")
        self.resize_and_save(wo_bg, f"{prefix}img_without_background.TIF")
        self.resize_and_save(bg, f"{prefix}background.TIF")

    def RB(self):
        app = MDApp.get_running_app()
        def threaded():
            app.start_loading()
            self.process_inputs()
            img1 = self.apply_bilateral_filter(app.img1)
            img2 = self.apply_bilateral_filter(app.img2)
            img1_scaled = self.resize_and_save(img1, "bil_scaled.TIF")
            img2_scaled = self.resize_and_save(img2, "sourcebil_scaled.TIF")
            merged1, wo_bg1, bg1 = self.process_lab(img1_scaled)
            merged2, wo_bg2, bg2 = self.process_lab(img2_scaled)
            self.save_processed_images("", merged1, wo_bg1, bg1)
            self.save_processed_images("source_", merged2, wo_bg2, bg2)
            app.stop_loading()
            Clock.schedule_once(lambda dt: app.root.get_screen("result").ids.rlt.reload())
        Thread(target=threaded, daemon=True).start()

class TemplateMatcher:
    def __init__(self, image_gray, source_dir, merge_crop_img):
        self.image_gray = image_gray
        self.source_dir = source_dir
        self.merge_crop_img = merge_crop_img
        self.detected_match = None

    def rotate_and_scale(self, img, angle, scale):
        rotated = imutils.rotate_bound(img, angle)
        resized = imutils.resize(rotated, width=int(rotated.shape[1] * scale))
        return rotated, resized

    def find_best_match(self):
        best_match = None
        for template_path in glob.glob(os.path.join(self.source_dir, '*.TIF')):
            template = cv2.imread(template_path, 0)
            th, tw = template.shape[:2]
            label = os.path.basename(template_path)
            for angle in np.arange(0, 360, 0.5):
                for scale in np.linspace(1.0, 0.2, 30):
                    rotated, resized = self.rotate_and_scale(self.image_gray, angle, scale)
                    if resized.shape[0] <= th or resized.shape[1] <= tw:
                        continue
                    result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(result >= 0.7)
                    for (x, y) in zip(loc[1], loc[0]):
                        match_val = result[y, x]
                        if not best_match or match_val > best_match['MATCH_VALUE']:
                            r = self.image_gray.shape[1] / float(resized.shape[1])
                            best_match = {
                                'IMAGE': rotated,
                                'TOP_LEFT_X': int(x * r),
                                'TOP_LEFT_Y': int(y * r),
                                'BOTTOM_RIGHT_X': int((x + tw) * r),
                                'BOTTOM_RIGHT_Y': int((y + th) * r),
                                'MATCH_VALUE': match_val,
                                'CHANGED_ANGLE': angle,
                                'SOURCE_NAME': label
                            }
        self.detected_match = best_match
        return best_match

    def crop_match_area(self):
        if not self.detected_match:
            return None
        match = self.detected_match
        cropped_img = match['IMAGE'][match['TOP_LEFT_Y']:match['BOTTOM_RIGHT_Y'],
                                     match['TOP_LEFT_X']:match['BOTTOM_RIGHT_X']]
        merge_rotated = imutils.rotate_bound(self.merge_crop_img, match['CHANGED_ANGLE'])
        cropped_merge = merge_rotated[match['TOP_LEFT_Y']:match['BOTTOM_RIGHT_Y'],
                                      match['TOP_LEFT_X']:match['BOTTOM_RIGHT_X']]
        cv2.imwrite(path_main + "beforecrop.TIF", match['IMAGE'])
        cv2.imwrite(path_main + "cropped.TIF", cropped_img)
        cv2.imwrite(path_main + "merged_crop.TIF", cropped_merge)
        return cropped_img

class PDFReport:
    def __init__(self, pdf_output_path):
        self.output_path = pdf_output_path
        self.elements = []
        self.styles = getSampleStyleSheet()

    def add_title_page(self, logo_path, project_title, authors):
        self.elements.append(Image(logo_path, 8 * inch, 4 * inch))
        self.elements.append(Spacer(1, 30))
        self.elements.append(Paragraph("Project Report", self.styles['Title']))
        self.elements.append(Spacer(1, 30))
        table_data = [['Name', 'Roll No.']] + authors
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ])
        table = Table(table_data)
        table.setStyle(table_style)
        self.elements.append(table)
        self.elements.append(PageBreak())

    def add_paragraph(self, text, style='Normal'):
        self.elements.append(Paragraph(text, self.styles[style]))
        self.elements.append(Spacer(1, 12))

    def add_image(self, img_path, caption, width=5*inch, height=7*inch):
        img = Image(img_path)
        img._restrictSize(width, height)
        self.elements.append(img)
        self.elements.append(Paragraph(f"<para align=center>{caption}</para>", self.styles['Heading4']))
        self.elements.append(Spacer(1, 12))

    def build_report(self):
        doc = SimpleDocTemplate(self.output_path, pagesize=A3)
        doc.build(self.elements)
