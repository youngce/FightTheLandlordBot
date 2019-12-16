import abc
import numpy as np
import Detector as d


class Area(abc.ABC):
    # np.arr
    def __init__(self, roi: d.ROI, original_img: np.array):
        self.roi = roi
        self.original_img = original_img

    def crop(self) -> np.array:
        return self.roi.crop(self.original_img)


# class PlayingArea(Area):


class CardArea(Area):
    def __init__(self, roi: d.ROI, original_img: np.array, corner_arg):
        super(CardArea, self).__init__(roi, original_img)
        # self.__init__(roi, original_img)
        self.corner_arg = corner_arg

    def detect(self):
        pass
