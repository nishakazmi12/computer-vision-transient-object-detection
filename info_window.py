# InfoWindow class and preprocessing logic

# Unified Transient Object Detection Script (Optimized)
# Contains: InfoWindow, TemplateMatcher, Comparator, PDFReport

import os
import cv2
import imutils
import glob
import numpy as np
import pandas as pd
import mahotas as mh
from threading import Thread
from matplotlib import pyplot as plt
from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock, mainthread
from cv2_rolling_ball import subtract_background_rolling_ball as sbrb
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

    def save_processed_images(self, prefix, img, img_wo_bg , bg):
        self.resize_and_save(img, f"{prefix}corrected_merge.TIF")
        self.resize_and_save(img_wo_bg, f"{prefix}img_without_background.TIF")
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

