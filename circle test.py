import numpy as np
import cv2

img = cv2.imread("four.jpg")
img = cv2.resize(img , (800 , 640))
image_copy = img.copy()
img = cv2.GaussianBlur(img , (7 , 7) , 3)

cv2.waitKey(1000)
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
ret , thresh = cv2.threshold(gray , 170 , 255 , cv2.THRESH_BINARY)

contours , _ = cv2.findContours(thresh , cv2.RETR_TREE , cv2.CHAIN_APPROX_NONE)
area = {}
for i in range(len(contours)):
    cnt = contours[i]
    ar = cv2.contourArea(cnt)
    area[i] = ar
print("Area", area)

srt = sorted(area.items() , key = lambda x : x[1] , reverse = True)
results = np.array(srt).astype("int")
print("Result",results)
num = np.argwhere(results[: , 1] > 500).shape[0]

for i in range(1 , num):
    image_copy = cv2.drawContours(image_copy , contours , results[i , 0] ,(0 , 255 , 0) , 3)
print("Number", num)
cv2.imshow("final" , image_copy)
# k = cv2.waitKey(10) & 0xFF  # "ESC exit"
# if k == 27:
#     break
#
# img.release()
# cv2.destroyAllWindows()

