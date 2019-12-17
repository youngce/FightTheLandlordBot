import abc
import numpy as np
import cv2
from typing import List
from typing import Mapping


class ROI:
    def __init__(self, x, y, width, height):
        self.x: int = int(x)
        self.y: int = int(y)
        self.width: int = int(width)
        self.height: int = int(height)

    def ys(self) -> list:
        return [self.y, (self.y + self.height)]

    def xs(self) -> list:
        return [self.x, self.x + self.width]

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, ROI):
            return self.__hash__() == other.__hash__()
        return False

    def __hash__(self):
        return hash((self.x, self.y, self.width, self.height))

    def crop(self, img):
        return img[self.y:self.y + self.height, self.x:self.x + self.width]

    def pt1(self):
        return self.x, self.y

    def pt2(self):
        return self.x + self.width, self.y + self.height

    def move(self, x: int, y: int) -> 'ROI':
        return ROI(self.x + x, self.y + y, self.width, self.height)


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


class Area(abc.ABC):
    # np.arr
    def __init__(self, roi: ROI, original_img: np.array):
        self.roi = roi
        self.original_img = cv2.resize(original_img, (1024, 658))
        self.img_cropped = roi.crop(self.original_img)


# class PlayingArea(Area):


class CardArea(Area):
    def __init__(self, roi: ROI, original_img: np.array, cs: CardSizes):
        super(CardArea, self).__init__(roi, original_img)
        # self.__init__(roi, original_img)
        self.cs = cs

    def find_rois_of_cards(self) -> List[ROI]:
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
            # width_of_card = self.cs.size_of_card.width
            if len(approx) == 4 and 9400 * 0.4 * 0.4 < area < 500000 * 0.4 * 0.4:
                def is_between(actual, expected, delta=0.1):
                    return expected * (1 - delta) < actual < expected * (1 + delta)

                x, y, w, h = cv2.boundingRect(cnt)
                # print(h, height_of_card)
                if is_between(h, height_of_card):
                    rois_selected.append(ROI(x, y, w, h))

        return rois_selected

    def find_rois_of_corners(self) -> Mapping[ROI, List[ROI]]:
        rois = self.find_rois_of_cards()
        rois_of_corners_by_roi_of_cards = dict()
        # Mapping[d.ROI, List[d.ROI]]
        size_of_card = self.cs.size_of_card
        size_of_corner = self.cs.size_of_corner
        width_without_corner = size_of_card.width - size_of_corner.width
        sum_of_margin = np.sum(self.cs.corner_of_margin)

        for roiOfCards in rois:
            print(roiOfCards.x, roiOfCards.y)
            num_of_cards = 0
            # int(round((roiOfCards.width - width_without_corner) / size_of_corner.width, 0))
            rois = []
            while True:
                x = roiOfCards.x + self.cs.size_of_corner.width * num_of_cards + self.cs.corner_of_margin[0]
                y = roiOfCards.y
                w = size_of_corner.width - sum_of_margin
                h = size_of_corner.height

                p1 = (x + 1, y + self.cs.height_of_rank + 1)
                p2 = (int(x + w / 2),
                      y + self.cs.height_of_rank + int((h - self.cs.height_of_rank) / 3))
                print(self.img_cropped[p1[1], p1[0]], self.img_cropped[p2[1], p2[0]])
                if not (self.img_cropped[p1[1], p1[0]] > 250 and self.img_cropped[p2[1], p2[0]] < 150):
                    break
                rois.append(ROI(x, y, w, h))
                num_of_cards += 1
            rois_of_corners_by_roi_of_cards.setdefault(roiOfCards, rois)
        return rois_of_corners_by_roi_of_cards

    def _debug(self, img):
        cv2.imshow("debug", img)
        cv2.waitKey(0)

    def show(self):

        rois_of_corners = self.find_rois_of_corners()
        bgr = cv2.cvtColor(self.img_cropped, cv2.COLOR_GRAY2BGR)
        for key in rois_of_corners.keys():
            cv2.rectangle(bgr, key.pt1(), key.pt2(), (255, 0, 0), 2)
            for v in rois_of_corners.get(key):
                # roi of rank
                rroi = ROI(v.x, v.y, v.width, self.cs.height_of_rank)
                # roi of suit
                sroi = ROI(v.x, v.y + self.cs.height_of_rank, v.width, v.height - self.cs.height_of_rank)

                p1 = (sroi.x + 1, sroi.y + 1)
                p2 = (int(sroi.x + sroi.width / 2), sroi.y + int(sroi.height / 3))
                # print(self.img_cropped[p1[1], p1[0]], self.img_cropped[p2[1], p2[0]])
                cv2.circle(bgr, p1, 1, (255, 255, 0), 1)
                cv2.circle(bgr, p2, 1, (255, 255, 0), 1)
                cv2.rectangle(bgr, rroi.pt1(), rroi.pt2(), (0, 255, 0), 1)
                cv2.rectangle(bgr, sroi.pt1(), sroi.pt2(), (0, 0, 255), 1)

        # cv2.resize(bgr,(bgr.shape[1]*2,bgr.shape[0]*2))
        self._debug(cv2.resize(bgr, (bgr.shape[1] * 2, bgr.shape[0] * 2)))

    def detect(self):
        pass


def own_cards_area(img):
    cardsize = Size(84, 112)
    cornersize = Size(36, 61)
    cs = CardSizes(cardsize, cornersize, (4, 8), 40)
    return CardArea(ROI(150, 440, 700, 130), img, cs)


def pool_cards_area(img):
    card = Size(68, 90)
    corner = Size(24, 50)
    cs = CardSizes(card, corner, (4, 2), 30)
    return CardArea(ROI(170, 170, 730, 160), img, cs)


def lord_cards_area(img):
    card = Size(58, 77)
    corner = Size(24, 42)
    cs = CardSizes(card, corner, (2, 2), 28)
    return CardArea(ROI(409, 47, 213, 89), img, cs)
