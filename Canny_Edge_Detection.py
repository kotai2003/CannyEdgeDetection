# --------------------------------------------------------
# Canny Edge Detection
# Tuning Program
# (c) 2021.07. S.Choe
# --------------------------------------------------------


import cv2
import numpy as np
from t_module import img_process

filePath = './image/image.png'
img_src = cv2.imread(filePath,1 )

height, width, channel = img_src.shape


# cv2 createTrackbar

def empty(x):
    pass


cv2.namedWindow('Parameters')
cv2.resizeWindow('Parameters', width = 600, height = 200)
cv2.createTrackbar('k_size_set', 'Parameters', 0, 10, empty)
cv2.createTrackbar('thresh_1', 'Parameters', 0, 500, empty)
cv2.createTrackbar('thresh_2', 'Parameters', 0, 500, empty)

# -----------------------------------------------
# Edge Detection Process
# 1. Color -> GrayScale
# 2. Gaussian Blur :: parameter -> kernel
# 3. Canny Edge Detection
# 3.1. Automatic Thresholds Finding
# 3.2. Quiita Thresholds Finding :: parameter -> threshold1, threshold2
# 4. Plotting
# -----------------------------------------------


while True:

    # 00. Make a Copy
    img_copy = img_src.copy()
    img_black = np.zeros_like(img_src)

    # 1. Color -> GrayScale
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Blur
    kernel = cv2.getTrackbarPos('k_size_set', 'Parameters')
    kernel = (kernel * 2) + 1

    img_blur = cv2.GaussianBlur(img_gray, (kernel, kernel), None)

    # 3. Canny Edge Detection

    # 3.1. Automatic Thresholds Finding
    med_val = np.median(img_blur)
    sigma = 0.33  # 0.33
    min_val = int(max(0, (1.0 - sigma) * med_val))
    max_val = int(max(255, (1.0 + sigma) * med_val))
    print('lower_val', min_val)
    print('upper_val', max_val)

    img_edge1 = cv2.Canny(img_blur, threshold1 = min_val, threshold2 = max_val)
    cv2.putText(img_edge1, 'th 1: {}'.format(str(min_val)), (0, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img_edge1, 'th 2: {}'.format(str(max_val)), (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img_edge1, 'k size: {}'.format('(' + str(kernel) + ',' + str(kernel) + ')'), (0, 150),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                (255, 255, 255), 1, cv2.LINE_AA)

    # 3.2 Quiita Thresholds Finding :: parameter -> threshold1, threshold2
    thres1_val = cv2.getTrackbarPos('thresh_1', 'Parameters')
    thres2_val = cv2.getTrackbarPos('thresh_2', 'Parameters')

    img_edge2 = cv2.Canny(img_blur, threshold1 = thres1_val, threshold2 = thres2_val)
    cv2.putText(img_edge2, 'th 1: {}'.format(str(thres1_val)), (0, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img_edge2, 'th 2: {}'.format(str(thres2_val)), (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img_edge2, 'ksize: {}'.format('(' + str(kernel) + ',' + str(kernel) + ')'), (0, 150),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Plotting

    img_stack = img_process.stackImages(0.5, ([[img_src, img_gray],
                                               [img_blur, img_black],
                                               [img_edge1, img_edge2]]))

    cv2.imshow('Result', img_stack)
    cv2.imshow('Canny', img_edge2)

    # qを押すと止まる。
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cv2.imwrite('image/automatic/image-canny-automatic.jpg', img_edge1)
cv2.imwrite('./image/image-canny-manual.jpg', img_edge2)
