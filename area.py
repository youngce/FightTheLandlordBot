import abc
import numpy as np
import Detector as d
import cv2
from typing import List
from typing import Mapping


class Area(abc.ABC):
    # np.arr
    def __init__(self, roi: d.ROI, original_img: np.array):
        self.roi = roi
        self.original_img = cv2.resize(original_img, (1024, 658))
        self.img_cropped = roi.crop(self.original_img)


# class PlayingArea(Area):
class Size:
    def __init__(self, w, h):
        self.width = w
        self.height = h


class CardSizes:

    def __init__(self, size_of_card: Size, size_of_corner: Size, corner_of_margin, height_of_rank):
        self.size_of_card = size_of_card
        self.size_of_corner = size_of_corner
        self.corner_of_margin = corner_of_margin
        self.height_of_rank = height_of_rank


class CardArea(Area):
    def __init__(self, roi: d.ROI, original_img: np.array, cs: CardSizes):
        super(CardArea, self).__init__(roi, original_img)
        # self.__init__(roi, original_img)
        self.cs = cs

    def find_rois_of_cards(self) -> List[d.ROI]:
        ret, thresh = cv2.threshold(self.img_cropped, 127, 255, 0)
        # cv2.imshow("thr", thresh)

        cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnts_sorted = sorted(cnts, key=lambda cnt: cv2.contourArea(cnt), reverse=True)
        rois_selected = []

        for cnt in cnts_sorted:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
            area = cv2.contourArea(cnt)
            # 9400 * 0.4 * 0.4 < area < 500000 * 0.4 * 0.4
            height_of_card = self.cs.size_of_card.height
            if len(approx) == 4 and 9400 * 0.4 * 0.4 < area < 500000 * 0.4 * 0.4:

                x, y, w, h = cv2.boundingRect(cnt)
                # print(h, height_of_card)
                if height_of_card * 0.9 < h < height_of_card * 1.1:
                    rois_selected.append(d.ROI(x, y, w, h))

        return rois_selected

    def find_rois_of_corners(self) -> Mapping[d.ROI, List[d.ROI]]:
        rois = self.find_rois_of_cards()
        rois_of_corners_by_roi_of_cards = dict()
        # Mapping[d.ROI, List[d.ROI]]
        size_of_card = self.cs.size_of_card
        size_of_corner = self.cs.size_of_corner
        width_without_corner = size_of_card.width - size_of_corner.width
        sum_of_margin = np.sum(self.cs.corner_of_margin)

        for roiOfCards in rois:

            num_of_cards = int(round((roiOfCards.width - width_without_corner) / size_of_corner.width, 0))
            rois = []
            for i in range(num_of_cards):
                x = roiOfCards.x + self.cs.size_of_corner.width * i + self.cs.corner_of_margin[0]
                y = roiOfCards.y
                w = size_of_corner.width - sum_of_margin
                h = size_of_corner.height
                rois.append(d.ROI(x, y, w, h))
            rois_of_corners_by_roi_of_cards.setdefault(roiOfCards, rois)
        return rois_of_corners_by_roi_of_cards

    def _debug(self, img):
        cv2.imshow("debug", img)
        cv2.waitKey(0)

    def show(self):

        rois_of_corners = self.find_rois_of_corners()
        bgr = cv2.cvtColor(self.img_cropped, cv2.COLOR_GRAY2BGR)
        for key in rois_of_corners.keys():
            cv2.rectangle(self.img_cropped, key.pt1(), key.pt2(), (255, 0, 0), 3)
            for v in rois_of_corners.get(key):
                # roi of rank
                rroi = d.ROI(v.x, v.y, v.width, self.cs.height_of_rank)
                # roi of suit
                sroi = d.ROI(v.x, v.y + self.cs.height_of_rank, v.width, v.height - self.cs.height_of_rank)
                cv2.rectangle(bgr, rroi.pt1(), rroi.pt2(), (0, 255, 0), 1)
                cv2.rectangle(bgr, sroi.pt1(), sroi.pt2(), (0, 0, 255), 1)

        # cv2.resize(bgr,(bgr.shape[1]*2,bgr.shape[0]*2))
        self._debug(cv2.resize(bgr, (bgr.shape[1] * 2, bgr.shape[0] * 2)))

    def detect(self):
        pass


class CardAreaFactory:
    @staticmethod
    def own(img):
        cardsize = Size(84, 112)
        cornersize = Size(36, 61)
        cs = CardSizes(cardsize, cornersize, (4, 8), 40)
        return CardArea(d.ROI(200, 440, 700, 130), img, cs)

    @staticmethod
    def pool(img):
        cardsize = Size(68, 90)
        cornersize = Size(24, 50)
        cs = CardSizes(cardsize, cornersize, (4, 2), 30)
        return CardArea(d.ROI(170, 170, 730, 160), img, cs)

    @staticmethod
    def lord(img):
        cardsize = Size(58, 77)
        cornersize = Size(24, 42)
        cs = CardSizes(cardsize, cornersize, (2, 2), 28)
        return CardArea(d.ROI(409, 47, 213, 89), img, cs)
