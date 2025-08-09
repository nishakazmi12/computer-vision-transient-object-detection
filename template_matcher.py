import glob
import os
import cv2
import imutils


path_main = "transient_object_detection_modular/staging"

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

