from detection import detect_road as rd
import cv2
import numpy as np
image = cv2.imread("train/images/66.jpg")
lines = rd(image)
# create a var img with same size as image
img = np.zeros_like(image)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
# Display the lines without image
cv2.imshow("Road Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()