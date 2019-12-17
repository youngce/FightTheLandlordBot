import abc
import numpy as np
import Detector as d
import cv2


class Area(abc.ABC):
    # np.arr
    def __init__(self, roi: d.ROI, original_img: np.array):
        self.roi = roi
        self.original_img = original_img
        self.img_cropped = roi.crop(original_img)


# class PlayingArea(Area):
class Size:
    def __init__(self, w, h):
        self.width = w
        self.height = h


class CardSizes:
    card_of_height = 0
    card_of_width = 0
    corner_of_width = 0
    corner_of_height = 0
    corner_of_margin = 0
    rank_of_height = 0

    def __init__(self, size_of_card: Size, corner_of_card: Size, corner_of_margin, rank_of_height):
        self.size_of_card = size_of_card
        self.corner_of_card = corner_of_card
        self.corner_of_margin = corner_of_margin
        self.rank_of_height = rank_of_height


class CardArea(Area):
    def __init__(self, roi: d.ROI, original_img: np.array, cardsizes: CardSizes):
        super(CardArea, self).__init__(roi, original_img)
        # self.__init__(roi, original_img)
        self.cardsizes = cardsizes

    def find_rois_of_cards(self) -> list:

        # img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        ret, thresh = cv2.threshold(self.img_cropped, 127, 255, 0)
        # cv2.imshow("thresh", thresh)
        # bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cntsSorted = sorted(cnts, key=lambda cnt: cv2.contourArea(cnt))
        rois_selected = []
        for cnt in cntsSorted:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
            area = cv2.contourArea(cnt)
            if len(approx) == 4 and 9400 < area < 500000:
                x, y, w, h = cv2.boundingRect(cnt)
                rois_selected.append(d.ROI(x, y, w, h))
                # print(area)
                # cv2.drawContours(bgr, [cnt], -1, (255, 0, 0), 3)
        return rois_selected

    def find_rois_of_corners(self) -> list:
        rois = self.find_rois_of_cards()
        for roiOfCards in rois:
            cropped = roiOfCards.crop(self.img_cropped)
            


    def show(self):
        pass


    def detect(self):
        pass
