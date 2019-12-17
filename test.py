import Detector as d
import area
import cv2

img = cv2.imread("/Users/mark/git/FightTheLandlordBot/mydata/rounds/xgtRSRUJJDMC35/1576165716809.jpg",
                 cv2.IMREAD_GRAYSCALE)

# bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# #
# ratio = 0.4
# w = int(img.shape[1] * ratio)
# h = int(img.shape[0] * ratio)
# print(w, h)
# cv2.resize(img, (w, h))
# cv2.imshow("img", cv2.resize(img, (w, h)))
# cv2.waitKey(0)

own = area.CardAreaFactory.pool(img)
own.show()
