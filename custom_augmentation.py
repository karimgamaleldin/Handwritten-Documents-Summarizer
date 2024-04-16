import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Custom transformation for Erosion
class Erosion(A.ImageOnlyTransform):
    def __init__(self, kernel_size=3, always_apply=False, p=0.5):
        super(Erosion, self).__init__(always_apply=always_apply, p=p)
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def apply(self, img, **params):
        return cv2.erode(img, self.kernel, iterations=1)

# Custom transformation for Dilation
class Dilation(A.ImageOnlyTransform):
    def __init__(self, kernel_size=3, always_apply=False, p=0.5):
        super(Dilation, self).__init__(always_apply=always_apply, p=p)
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def apply(self, img, **params):
        return cv2.dilate(img, self.kernel, iterations=1)
