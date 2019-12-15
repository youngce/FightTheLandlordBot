import numpy as np
import recognizer
import cv2


class CornerImg:
    def __init__(self, img: np.array, arg):

        self.img = img
        self.width = img.shape[0]
        self.height = img.shape[1]
        self.rank_height = arg.rank_height

    def resize(self, img, is_rank: bool = True):
        # rankCropped = cv2.resize(rankROI.crop(img), (40, 25))
        #     suitCropped = cv2.resize(suitROI.crop(img),(40,40))
        size = (40, 25)
        if not is_rank:
            size = (40, 40)
        return cv2.resize(img, size)

    def rank_img(self):
        return self.resize(self.img[0:self.rank_height, 0:self.width])

    def suit_img(self):
        return self.resize(self.img[self.rank_height:self.height, 0:self.width], False)

    def isCard(self):
        cv2.imshow("rank", self.rank_img())
        cropped = recognizer.crop_maximum_area_of_contour(self.rank_img())
        cv2.imshow("cropped", cropped)
        minimum_rank_diff = 100000
        rank_name = "Unknown"
        for trank in recognizer.tranks:
            diff_img = cv2.absdiff(cv2.resize(cropped, (trank.img.shape[1], trank.img.shape[0])), trank.img)
            rank_diff = int(np.sum(diff_img) / 255)

            if minimum_rank_diff > rank_diff:
                minimum_rank_diff = rank_diff
                rank_name = trank.name
                # cv2.imshow(trank.name, diff_img)
                # cv2.waitKey(0)

        print(rank_name,minimum_rank_diff)
