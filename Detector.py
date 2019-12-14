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
        cv2.rectangle(img, (roi.x, roi.y), (roi.x + roi.weight, roi.y + roi.high), (255, 0, 0), 3)
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


# file="/Users/mark/git/FightTheLandlordBot/mydata/rounds/AyaiGREGJJDMC29/1576164651610.jpg"
# file = "/Users/mark/git/FightTheLandlordBot/mydata/rounds/dsieGECJJDMC10/cards.jpg"
file = "/Users/mark/git/FightTheLandlordBot/mydata/rounds/AyaiGREGJJDMC29/1576164658135_copy.jpg"
# file = "/Users/mark/git/FightTheLandlordBot/mydata/rounds/AyaiGREGJJDMC29/1576164658135.jpg"

img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

# x, y, w, h = cv2.selectROI("cards", img)
# print(x, y, w, h)
# cropped = img[y:y + h, x:x + w]
# gray = cropped
bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
gray = img
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find Contour
cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
index_sort = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)
num_of_cards = 0
print("num of cnts: %s" % len(cnts))
CORNER_HEIGHT = 155
CORNER_WEIGHT = 90
CARD_WEIGHT = 208
RANK_HEIGHT = 104
cnts_selected = []
train_ranks = load_ranks("./cards/")

# def boundingRect_of_maxContour(img):
#     edged = cv2.Canny(img, 30, 300)
#     cnts, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # print(len(cnts))
#     maxCnt = sorted(cnts, key=lambda cnt: cv2.contourArea(cnt), reverse=True)[0]
#     # cv2.imshow("test",edged)
#     # cv2.drawContours(img,[maxCnt],-1, (255, 0, 0),3)
#     x, y, w, h = cv2.boundingRect(maxCnt)
#     return edged[y:y + h, x:x + w]


for i in index_sort:
    area = cv2.contourArea(cnts[i])
    cnt = cnts[i]
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
    if area > 57000 and area < 460000 and len(approx) == 4:
        print("area of cnt is selected: %s" % cv2.contourArea(cnts[i]))
        cnts_selected.append(cnts[i])
        x, y, w, h = cv2.boundingRect(cnt)
        print("weight: %s, hight: %s" % (w, h))
        num_of_cards = int(round(w - 208) / 90 + 1)
        print("num of cards is %s" % num_of_cards)
        # cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for j in range(num_of_cards):
            rank_img = gray[(y + RANK_HEIGHT):(y + CORNER_HEIGHT),
                       (x + CORNER_WEIGHT * j):(x + CORNER_WEIGHT * (j + 1))]
            # rank_img = boundingRect_of_maxContour(rank_img)

            cv2.imshow("wind",rank_img)
            # eg = cv2.Cann
            # y(rank_img, 30, 300)
            cv2.imwrite("./cards/suits%s.jpg" % j, rank_img)
            # best_rank_match_diff = 10000
            # best_rank_name = "Unknown"
            #
            # for tr in train_ranks:
            #     weight = tr.img.shape[1]
            #     height = tr.img.shape[0]
            #     # print("old size: $s"%rank_img.shape)
            #     resized1 = cv2.resize(rank_img, (weight, height))
            #     # resized2 = cv2.resize(tr.img, (0, 0), fx=4, fy=4)
            #     diff_img = cv2.absdiff(resized1, tr.img)
            #
            #     rank_diff = int(np.sum(diff_img) / 255)
            #
            #     # print("rank: %s, diff: %s"%(tr.name,rank_diff))
            #     if best_rank_match_diff > rank_diff:
            #         best_rank_match_diff = rank_diff
            #         best_rank_name = tr.name
            #
            #
            # print(best_rank_name)

            # cv2.rectangle(bgr, (x + CORNER_WEIGHT * j, y), (x + CORNER_WEIGHT * (j + 1), int(y + RANK_HEIGHT)),
            #               (0, 255, 0),
            #               2)
            # cv2.rectangle(bgr, (x + CORNER_WEIGHT * j, y),
            #               (x + CORNER_WEIGHT * (j + 1), y + CORNER_HEIGHT), (255, 0, 0),
            #               2)

# 1 card's weight is 208
# 2 card's weight is 298
#
# 26888 half card
# 58398 whole card
print("num of cnts are selectd: %s" % len(cnts_selected))
# cv2.drawContours(bgr, cnts_selected, -1, (0, 255, 0), 3)
# cv2.drawContours(gray, cnts_selected, -1, (0, 255, 0), 3)
# cv2.imshow('mask', gray)
# cv2.imshow("cards", bgr)
# r = cv2.selectROI("cards", img)
# print(r)
# y = r[1]
# x = r[0]
# h = r[3]
# w = r[2]
# cropped = img[y:y + h, x:x + w]

# cnts, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# print("num of cnts: %s" % len(cnts))
# for cnt in cnts:
#     print("area of cnt: %d" % cv2.contourArea(cnt))
#
# cv2.drawContours(cropped, cnts, -1, (255, 0, 0),3)
# # cv2.imshow("cropped", cropped)
#
# cv2.drawContours(img, cnts, -1, (0, 0, 255),3)
# cv2.imshow("cropped", img)
# # roundId=Detector.get_round_id(img)
# # print(roundId)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
