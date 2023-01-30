import cv2 as cv
import numpy as np

def on_trackbar(val):
    threshold = val
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(gray_image, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    dilation = cv.dilate(closing,kernel,iterations = 1)
    erosion = cv.erode(dilation,kernel,iterations = 1)
    #blackhat = cv.morphologyEx(gray_image, cv.MORPH_BLACKHAT, kernel)
    noise = cv.fastNlMeansDenoising(erosion, None, 15, 7, 21)
    
    _, thresholded = cv.threshold(noise, threshold, 255, cv.THRESH_BINARY)

    dilation = cv.dilate(thresholded,kernel,iterations = 2)
    erosion = cv.erode(dilation,kernel,iterations = 2)
    cv.imshow("Thresholded Image", erosion)

# Load the grayscale image
gray_image = cv.imread("C:\\Users\\karti\\OneDrive\\Desktop\\Project\\CV Assignment\imgd.BMP", cv.IMREAD_GRAYSCALE)
gray_image = cv.resize(gray_image,(800,600))

# Create a window
cv.namedWindow("Thresholded Image")

# Create a trackbar
threshold = 128
cv.createTrackbar("Threshold", "Thresholded Image", threshold, 255, on_trackbar)


# Show the image
on_trackbar(threshold)
cv.waitKey(0)
cv.destroyAllWindows()
