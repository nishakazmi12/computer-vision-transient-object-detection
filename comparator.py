# comparator.py
import cv2
import numpy as np
import os

result = "transient_object_detection_modular"

class Comparator:
    def __init__(self, test_img_path, reference_img_path):
        self.test_img_path = test_img_path
        self.reference_img_path = reference_img_path
        #self.detected_output_path = os.path.join(os.path.dirname(test_img_path), "DetectedObjects.TIF")
        #self.reference_output_path = os.path.join(os.path.dirname(test_img_path), "DetectedObjects_Source.TIF")
        self.detected_output_path = result + "/DetectedObjects.TIF"
        self.reference_output_path = result + "/DetectedObjects_Source.TIF"

    def compare_images(self):
        # Load both images in grayscale
        img_test = cv2.imread(self.test_img_path, cv2.IMREAD_GRAYSCALE)
        img_ref = cv2.imread(self.reference_img_path, cv2.IMREAD_GRAYSCALE)

        if img_test is None or img_ref is None:
            raise FileNotFoundError("One or both images could not be loaded.")

        # Ensure both images are the same size
        if img_test.shape != img_ref.shape:
            img_ref = cv2.resize(img_ref, (img_test.shape[1], img_test.shape[0]))

        # Compute absolute difference
        diff = cv2.absdiff(img_ref, img_test)

        # Threshold the difference
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        # Find contours of different objects
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw detections on both images
        detected_img_test = cv2.cvtColor(img_test, cv2.COLOR_GRAY2BGR)
        detected_img_ref = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR)

        for cnt in contours:
            if cv2.contourArea(cnt) > 5:  # filter small noise
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(detected_img_test, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.rectangle(detected_img_ref, (x, y), (x + w, y + h), (0, 0, 255), 1)

        # Save output images
        cv2.imwrite(self.detected_output_path, detected_img_test)
        cv2.imwrite(self.reference_output_path, detected_img_ref)

        print(f"Saved detected objects image at: {self.detected_output_path}")
        print(f"Saved reference objects image at: {self.reference_output_path}")

