import cv2
import numpy as np


def detect_road(image):
    # Define a threshold for how close the R, G, and B values must be
    threshold = 50

    # Find the gray regions
    gray_regions = np.abs(image[:,:,0] - image[:,:,1]) < threshold
    gray_regions &= np.abs(image[:,:,0] - image[:,:,2]) < threshold
    gray_regions &= np.abs(image[:,:,1] - image[:,:,2]) < threshold

    # Convert the boolean mask to uint8
    gray_regions_uint8 = (gray_regions * 255).astype(np.uint8)
    # 进行Canny边缘检测
    cv2.Canny(gray_regions_uint8, 100, 250)
    # 提取面积最大的轮廓并用四边形将轮廓包围
    contours, _ = cv2.findContours(gray_regions_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    # 使用四边形创建roi
    mask = np.zeros_like(gray_regions_uint8)
    cv2.fillPoly(mask, [approx], 255)

    # 在roi内识别道路边缘
    edges = cv2.Canny(image, 100, 250)
    edges = cv2.bitwise_and(edges, mask)
    # 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=50)
    # only return lines detected in the road region
    return lines
if __name__ == "__main__":
    # read all images from "train/images" folder and display them with a road detection overlay and 0.1s gap
    import os
    for filename in os.listdir("train/images"):
        image = cv2.imread("train/images/" + filename)
        lines = detect_road(image)
        img = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.imshow("Road Detection", img)
            cv2.waitKey(100)
        cv2.imwrite("train/" + filename, img)    
    cv2.destroyAllWindows()