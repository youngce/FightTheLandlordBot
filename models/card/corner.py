import numpy as np
import recognizer
import cv2


class CornerImg:
    width = 40
    height = 65
    rank_height = 40

    def __init__(self, img: np.array, arg):
        self.img = cv2.resize(img, (40, 65))
        #
        # self.width = img.shape[0]
        # self.height = img.shape[1]
        # self.rank_height = arg.rank_height

    def rank_img(self):
        return self.img[0:self.rank_height, 0:self.width]

    def suit_img(self):
        # cv2.imshow("suit",self.img[self.rank_height:self.height, 0:self.width])
        # cv2.waitKey(0)
        return self.img[self.rank_height:self.height, 0:self.width]

    def mse(self, imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err

    @staticmethod
    def show(img):
        cv2.imshow("show", img)
        cv2.waitKey(0)

    def isCard(self):
        suit = self.suit_img()
        x1, y1 = (1, 1)
        x2, y2 = (int(suit.shape[1] / 2), int((suit.shape[0]) / 3 + 1))

        print(suit[y2, x2], suit[y1, x1])
        brg = cv2.cvtColor(suit, cv2.COLOR_GRAY2BGR)
        cv2.circle(brg, (x2, y2), 1, (255, 0, 0), 1)
        cv2.circle(brg, (1, 1), 1, (255, 0, 0), 1)
        self.show(brg)
        return suit[y2, x2] < 100 and suit[y1, x1] > 250
        #
        # if np.any() != 255:
        #     print("the pixel is not white")
        #
        # if np.any(suit[1, 1]) == 255:
        #     print("the pixel is white")
        # # brg=cv2.cvtColor(suit,cv2.COLOR_GRAY2BGR)
        # cv2.circle(suit, (x2, y2), 1, (255, 0, 0), 1)
        # cv2.circle(suit, (1, 1), 1, (255, 0, 0), 1)
        # self.show(suit)
        # cropped = recognizer.crop_maximum_area_of_contour(self.rank_img())
        # # cv2.imshow("cropped", cropped)
        # minimum_rank_diff = 100000
        # rank_name = "Unknown"
        # for trank in recognizer.tranks:
        #     # resized = cv2.resize(cropped, (trank.img.shape[1], trank.img.shape[0]))
        #     # diff_img = cv2.absdiff(resized, trank.img)
        #
        #     mse = self.mse(cv2.resize(cropped, (trank.img.shape[1], trank.img.shape[0])), trank.img)
        #     # mse = int(np.sum(np.square(diff_img)) / 255)
        #     # cv2.imshow(trank.name, trank.img)
        #     # cv2.waitKey(0)
        #     # print(trank.name, mse)
        #     if minimum_rank_diff > mse:
        #         minimum_rank_diff = mse
        #         rank_name = trank.name
        #     # cv2.imshow(trank.name, trank.img)
        #
        # print(rank_name, minimum_rank_diff)
        # cv2.waitKey(0)
