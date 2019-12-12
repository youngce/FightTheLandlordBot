import cv2
import numpy as np
import pytesseract
from PIL import Image


class Detector:

    # def __init__(self, img: np.ndarray):
    #     self.img: np.ndarray = img
    @staticmethod
    def get_round_id(img: np.ndarray) -> str:
        roi = ROI(216, 27, 259, 34)
        img_cropped = img[roi.ys()[0]:roi.ys()[1], roi.xs()[0]:roi.xs()[1]]

        return pytesseract.image_to_string(Image.fromarray(img_cropped), lang="eng")


class ROI:
    def __init__(self, x, y, weight, high):
        self.x: int = x
        self.y: int = y
        self.weight: int = weight
        self.high: int = high

    def ys(self) -> list:
        return [self.y, (self.y + self.high)]

    def xs(self) -> list:
        return [self.x, self.x + self.weight]
