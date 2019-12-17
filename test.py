import Detector as d
import area
import cv2

path = "mydata/rounds/AyaiGREGJJDMC29/1576164651610.jpg"
img = cv2.imread(path,
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

own = area.own_cards_area(img)
own.show()
