import cv2
import numpy as np
import pytesseract
from PIL import Image


class Detector:

    # def __init__(self, img: np.ndarray):
    #     self.img: np.ndarray = img
    @staticmethod
    def get_round_id(img: np.ndarray) -> str:
        roi = ROI(210, 25, 300, 40)
        img_cropped = img[roi.ys()[0]:roi.ys()[1], roi.xs()[0]:roi.xs()[1]]
        cv2.rectangle(img, (roi.x,roi.y), (roi.x+roi.weight,roi.y+roi.high), (255, 0, 0), 3)
        # cv2.imshow("full",img)
        # cv2.imshow("cropped", img_cropped)


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

# img=cv2.imread("./mydata/images/test2.png",cv2.IMREAD_GRAYSCALE)
# roundId=Detector.get_round_id(img)
# print(roundId)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
