from PIL import Image
import cv2
import numpy as np
import pytesseract

img = cv2.imread("../../Downloads/AcetoFive.JPG")

# img = cv2.imread("../../Downloads/20131030153346984.jpg")
# text = pytesseract.image_to_string(im)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
img_w, img_h = np.shape(img)[:2]
bkg_level = gray[int(img_h / 100)][int(img_w / 2)]
thresh_level = bkg_level + 60
print(img_w,img_h,thresh_level)
ret, binary = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)

cnts, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)
print(index_sort)

cnts_sort = []
hier_sort = []
cnt_is_card = np.zeros(len(cnts),dtype=int)
for i in index_sort:
    cnts_sort.append(cnts[i])
    hier_sort.append(hier[0][i])
# cv2.drawContours(img, cnts, -1, (0, 0, 255), 3)

# cv2.imwrite("./img2.jpg", img)
