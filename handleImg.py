import matplotlib.pyplot as plt
import numpy as np
import cv2
img=cv2.imread("./test.png")

# r = 1
#
# fig, ax = plt.subplots()
# ax.imshow(img, extent=(0,img.shape[1]/r,0,img.shape[0]/r) )
# ax.set_xlabel("distance [m]")
# ax.set_ylabel("distance [m]")
#
# plt.show()
# r=cv2.selectROI(img)
r=[216, 27, 259, 34]
# print(r)
# cv2.rectangle(img,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,255,0),5)
imCrop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
# cv2.imwrite("./round.png",imCrop)
# Display cropped image
cv2.imshow("round.png", imCrop)
# cv2.imshow("rec",img)

import pytesseract
from PIL import Image
# pImg.fromarray(imCrop)
# edges = cv2.Canny(imCrop,100,200)

roundImg=Image.fromarray(imCrop)
# cv2.imread("round.png_screenshot_11.12.2019.png")
res = pytesseract.image_to_string(roundImg,lang="eng")
print("res: "+res)
cv2.waitKey(0)
cv2.destroyAllWindow()
