import cv2
import numpy as np
from PIL import Image, ImageFilter
thresh1= 128
img = cv2.imread("police.png",0)
#revtal, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_OTSU)
thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    if cv2.contourArea(c) < 10:
        cv2.drawContours(thresh, [c], -1, (0,0,0), -1)

result = 255 - thresh

img_binary = cv2.threshold(result, thresh1, 255, cv2.THRESH_BINARY)[1]
img_not= cv2.bitwise_not(img_binary)

cv2.imwrite("polinv.png",img_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()