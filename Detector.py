import cv2
import numpy as np
import pytesseract
from PIL import Image
from models.card import argument
from models.card.argument import CornerArgumentFactory
from models.card.corner import CornerImg


class Detector:

    # def __init__(self, img: np.ndarray):
    #     self.img: np.ndarray = img
    @staticmethod
    def get_round_id(img: np.ndarray) -> str:
        roi = ROI(210, 25, 300, 40)
        img_cropped = img[roi.ys()[0]:roi.ys()[1], roi.xs()[0]:roi.xs()[1]]
        cv2.rectangle(img, (roi.x, roi.y), (roi.x + roi.width, roi.y + roi.height), (255, 0, 0), 3)
        # cv2.imshow("full",img)
        # cv2.imshow("cropped", img_cropped)

        return pytesseract.image_to_string(Image.fromarray(img_cropped), lang="eng")

    @staticmethod
    def find_rois_of_whole_cards(img: np.ndarray, roiSelected=None):
        cropped = img

        edged = cv2.Canny(cropped, 30, 100)
        # edged = img

        # r=0.1
        # w= int(img.shape[0]*r)
        # h= int(img.shape[1]*r)
        # edged = cv2.GaussianBlur(img, (w, h), cv2.BORDER_DEFAULT)

        cnts_selected = []
        cnts, hier = cv2.findContours(edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        index_sort = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)
        rois = []

        for i in index_sort:
            area = cv2.contourArea(cnts[i])
            cnt = cnts[i]
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
            # area=57227,28000
            x, y, w, h = cv2.boundingRect(cnt)
            if len(approx) == 4 and 28000 < area < 58000 and w / h < 1:
                cnts_selected.append(cnts[i])
                rois.append(ROI(x, y, w, h))

        return list(set(rois))

    @staticmethod
    def find_corner_of_cards(img: np.array, rois_of_cards, arg: argument) -> list:

        cornerROIs = []
        for roi in rois_of_cards:

            for i in range(20):
                w = arg.width
                wm = arg.width_margin
                s = arg.shift
                h = arg.height
                hm = arg.height_margin
                cornerROI = ROI(roi.x - wm * (i - 1) - (w + wm) * i + s, roi.y + hm, w, h)
                if cornerROI.crop(img).size == 0 or not CornerImg(cornerROI.crop(img), arg).isCard():
                    break

                cornerROIs.append(cornerROI)

            # for i in range(num):
            #     w = arg.width
            #     wm = arg.width_margin
            #     s = arg.shift
            #     h = arg.height
            #     hm = arg.height_margin
            #     # arg.height_margin
            #     # rh = arg.rank_height
            #
            #     cornerROI = ROI(roi.x - wm * (i - 1) - (w + wm) * i + s, roi.y + hm, w, h)
            #     # suitROI = cornerROI.move(0, rh)
            #     cornerROIs.append(cornerROI)
            # cv2.rectangle(bgr, cornerROI.pt1(), cornerROI.pt2(), (0, 0, 255), 1)
            # cv2.rectangle(bgr, suitROI.pt1(), suitROI.pt2(), (0, 255, 0), 1)

        return cornerROIs


# cv2.imshow("test", bgr)
# cv2.waitKey(0)


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

    # def range_of_y(self):
    #     return range(self.y, self.y + self.height)
    #
    # def range_of_x(self):
    #     return range(self.x, self.x + self.width)

    def crop(self, img):
        return img[self.y:self.y + self.height, self.x:self.x + self.width]

    def pt1(self):
        return self.x, self.y

    def pt2(self):
        return self.x + self.width, self.y + self.height

    def move(self, x: int, y: int) -> 'ROI':
        return ROI(self.x + x, self.y + y, self.width, self.height)


class Train_ranks:
    """Structure to store information about train rank images."""

    def __init__(self):
        self.img = []  # Thresholded, sized rank image loaded from hard drive
        self.name = "Placeholder"


def load_ranks(filepath):
    """Loads rank images from directory specified by filepath. Stores
    them in a list of Train_ranks objects."""

    train_ranks = []
    i = 0

    ranks = ['A', '2', '3', '4', '5', '6', '7',
             '8', '9', '10', 'J', 'Q', 'K', 'JOKER']
    # ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven',
    #  'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King']
    for Rank in ranks:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.jpg'
        print(filename)

        edged = cv2.Canny(cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE), 30, 300)

        train_ranks[i].img = edged
        i = i + 1

    return train_ranks


file = "test/ownCards.jpg"
# file = "test/ownCards2.jpg"
# numOfCards = 17
# file = "test/ownCards3.jpg"
# numOfCards = 4
cornerArg = CornerArgumentFactory.own()

# file = "test/leftcards.jpg"

# file = "test/lordcard.jpg"
# numOfCards = 3
# cornerArg = CornerArgumentFactory.lord_cards()

# file = "test/pool1.jpg"
# numOfCards = 4
# file = "test/pool2.jpg"
# numOfCards=2
# file = "test/pool3.jpg"
# numOfCards = 10
# file = "test/pool4.jpg"
# numOfCards = 1
# cornerArg = CornerArgumentFactory.pool()

img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
print(img[0, 0])

bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

rois = Detector.find_rois_of_whole_cards(img)
c = 100
for roi in rois:
    cv2.rectangle(bgr, roi.pt1(), roi.pt2(), (c, 0, 0), 2)
    c += 100

cornerROIs = Detector.find_corner_of_cards(img, rois, cornerArg)

print("num of card: %s, num of corner: %s" % (len(rois), len(cornerROIs)))

i = 0
from models.card.corner import CornerImg

for roi in cornerROIs:
    # if i != 6:
    #     i += 1
    #     continue

    # cv2.rectangle(bgr, roi.pt1(), roi.pt2(), (255, 0, 0), 1)
    rankROI = ROI(roi.x, roi.y, roi.width, cornerArg.rank_height)
    suitROI = ROI(roi.x, roi.y + cornerArg.rank_height, roi.width, roi.height - cornerArg.rank_height)
    # cv2.rectangle(bgr, rankROI.pt1(), rankROI.pt2(), (0, 0, 255), 1)
    # cv2.rectangle(bgr, suitROI.pt1(), suitROI.pt2(), (0, 255, 0), 1)
    centerX = (suitROI.pt1()[0] + suitROI.pt2()[0]) / 2
    centerY = (suitROI.pt1()[1] + suitROI.pt2()[1]) / 2 - 10
    # cv2.circle(bgr, (int(centerX), int(centerY)), 1, (255, 0, 0), 2)
    # cv2.circle(bgr, suitROI.pt1(), 1, (255, 0, 0), 1)
    # cv2.waitKey(0)
    cornerCropped = roi.crop(img)

    ci = CornerImg(cornerCropped, cornerArg)
    ci.isCard()
    i += 1
    # rankCropped = cv2.resize(rankROI.crop(img), (40, 25))
    # suitCropped = cv2.resize(suitROI.crop(img), (40, 40))

    # OWN
    # rank width: 82 height: 56
    # suit width: 53 height: 56
    # lord
    # rank width: 55 height: 40
    # suit width: 37 height: 40
    #
    # print("rank width: %s height: %s" % (rankCropped.shape[0], rankCropped.shape[1]))
    # print("suit width: %s height: %s" % (suitCropped.shape[0], suitCropped.shape[1]))
    # cv2.imwrite("cards/rank-%i.jpg" % i, rankCropped)
    # cv2.imwrite("cards/suit-%i.jpg" % i, suitCropped)
    i += 1
#
# cv2.imshow("bgr", bgr)
# cv2.waitKey(0)

# edged = cv2.Canny(img, 0, 500)
# cv2.imshow("edged", edged)
# # x, y, w, h = cv2.selectROI("cards", img)
# # print(x, y, w, h)
# # cropped = img[y:y + h, x:x + w]
# # gray = cropped
# bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# gray = edged
# ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# #
# # Find Contour
# cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# index_sort = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)
# num_of_cards = 0
# print("num of cnts: %s" % len(cnts))
# CORNER_HEIGHT = 155
# CORNER_WEIGHT = 90
# CARD_WEIGHT = 208
# RANK_HEIGHT = 104
# cnts_selected = []
# train_ranks = load_ranks("./cards/")
#
# # def boundingRect_of_maxContour(img):
# #     edged = cv2.Canny(img, 30, 300)
# #     cnts, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# #     # print(len(cnts))
# #     maxCnt = sorted(cnts, key=lambda cnt: cv2.contourArea(cnt), reverse=True)[0]
# #     # cv2.imshow("test",edged)
# #     # cv2.drawContours(img,[maxCnt],-1, (255, 0, 0),3)
# #     x, y, w, h = cv2.boundingRect(maxCnt)
# #     return edged[y:y + h, x:x + w]
#
#
# for i in index_sort:
#     area = cv2.contourArea(cnts[i])
#     cnt = cnts[i]
#     peri = cv2.arcLength(cnt, True)
#     approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
#     # area > 57000 and area < 460000
#     cv2.imshow("cnts", bgr)
#     if len(approx) > 0:
#         print("area of cnt is selected: %s" % cv2.contourArea(cnts[i]))
#         cnts_selected.append(cnts[i])
#         x, y, w, h = cv2.boundingRect(cnt)
#         print("weight: %s, hight: %s" % (w, h))
#         num_of_cards = int(round(w - 208) / 90 + 1)
#         print("num of cards is %s" % num_of_cards)
#         print("area: %s, length of approx:%s" % (area, len(approx)))
#
#         cv2.drawContours(bgr, [cnts[i]], -1, (255, 0, 0), 3)
#         cv2.waitKey(0)
#
# #         # cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
# #         for j in range(num_of_cards):
# #             # rank_img = gray[(y + RANK_HEIGHT):(y + CORNER_HEIGHT),
# #             #            (x + CORNER_WEIGHT * j):(x + CORNER_WEIGHT * (j + 1))]
# #
# #             rank_img = gray[y :(y + RANK_HEIGHT),
# #                        (x + CORNER_WEIGHT * j):(x + CORNER_WEIGHT * (j + 1))]
# #             # rank_img = boundingRect_of_maxContour(rank_img)
# #
# #             cv2.imshow("wind",rank_img)
# #
# #             best_rank_match_diff = 10000
# #             best_rank_name = "Unknown"
# #
# #             for tr in train_ranks:
# #                 weight = tr.img.shape[1]
# #                 height = tr.img.shape[0]
# #                 # print("old size: $s"%rank_img.shape)
# #                 resized1 = cv2.resize(rank_img, (weight, height))
# #                 # resized2 = cv2.resize(tr.img, (0, 0), fx=4, fy=4)
# #                 diff_img = cv2.absdiff(resized1, tr.img)
# #
# #                 rank_diff = int(np.sum(diff_img) / 255)
# #
# #                 # print("rank: %s, diff: %s"%(tr.name,rank_diff))
# #                 if best_rank_match_diff > rank_diff:
# #                     best_rank_match_diff = rank_diff
# #                     best_rank_name = tr.name
# #
# #
# #             print(best_rank_name)
# #
# #             # cv2.rectangle(bgr, (x + CORNER_WEIGHT * j, y), (x + CORNER_WEIGHT * (j + 1), int(y + RANK_HEIGHT)),
# #             #               (0, 255, 0),
# #             #               2)
# #             # cv2.rectangle(bgr, (x + CORNER_WEIGHT * j, y),
# #             #               (x + CORNER_WEIGHT * (j + 1), y + CORNER_HEIGHT), (255, 0, 0),
# #             #               2)
# #
# # # 1 card's weight is 208
# # # 2 card's weight is 298
# # #
# # # 26888 half card
# # # 58398 whole card
# # print("num of cnts are selectd: %s" % len(cnts_selected))
# # # cv2.drawContours(bgr, cnts_selected, -1, (0, 255, 0), 3)
# # # cv2.drawContours(gray, cnts_selected, -1, (0, 255, 0), 3)
# # # cv2.imshow('mask', gray)
# # # cv2.imshow("cards", bgr)
# # # r = cv2.selectROI("cards", img)
# # # print(r)
# # # y = r[1]
# # # x = r[0]
# # # h = r[3]
# # # w = r[2]
# # # cropped = img[y:y + h, x:x + w]
# #
# # # cnts, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# # # print("num of cnts: %s" % len(cnts))
# # # for cnt in cnts:
# # #     print("area of cnt: %d" % cv2.contourArea(cnt))
# # #
# # # cv2.drawContours(cropped, cnts, -1, (255, 0, 0),3)
# # # # cv2.imshow("cropped", cropped)
# # #
# # # cv2.drawContours(img, cnts, -1, (0, 0, 255),3)
# # # cv2.imshow("cropped", img)
# # # # roundId=Detector.get_round_id(img)
# # # # print(roundId)
# # cv2.destroyAllWindows()
