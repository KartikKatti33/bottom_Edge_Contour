# Author: Kartik A Katti

import cv2 as cv
import numpy as np
import sys

class BottomEdgeMapper:
    
    def __init__(self, image_path):
        #Read the image, resize it and make a copy for reference

        try:
            #Exception handling to read the image
            self.image = cv.imread(image_path)
        except Exception as reason:
            print("There was an Exception reading the image with reason \n", reason)
            sys.exit(1)

        self.image = cv.resize(self.image,(800,600))
        self.image_copy = self.image.copy()
        cv.imshow("self.image",self.image)

    def GrayscaleConverter(self, image):
        #Pure function to convert color to grayscale image
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    #Pure Functions for Morphological Operations
    def MorphologicalOperation_Opening(self, image, kernel):
        return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

    def MorphologicalOperation_Closing(self, image, kernel):
        return cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)

    def MorphologicalOperation_Dialate(self, image, kernel, iterations):
        return cv.dilate(image,kernel,iterations = iterations)

    def MorphologicalOperation_Erode(self, image, kernel, iterations):
        return cv.erode(image,kernel,iterations = iterations)

    def DeNoiser(self, image):
        #Pure function to perform DeNoising
        # Tried various option like Gaussian Blur, Median Blur, Bilateral Filter etc 
        # but fastNlMeansDenoising gave best result
        
        return cv.fastNlMeansDenoising(image, None, 15, 7, 21)

    def Threshold_Image(self, image, lower_threshold, upper_threshold):
        ret, thresh1 = cv.threshold(image,lower_threshold,upper_threshold,cv.THRESH_BINARY)
        return thresh1

    def CannyEdgeDetector(self, image, threshold1, threshold2, apertureSize, L2gradient):
        #Pure Function for Canny Edge Detection Algorithm with parameters
        return cv.Canny(image, threshold1, threshold2, apertureSize=apertureSize, L2gradient=L2gradient)

    def Contours(self, image):
        # detect the contours on the binary image using cv.CHAIN_APPROX_NONE
        contours, hierarchy = cv.findContours(image, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
        return contours

    def LongestContoursFilter(self, contours):
        # Create a list to store the contours and their lengths
        contours_lengths = []

        # calculate the length of contour arc to draw only few long contour and avoid unwanted noise
        # Loop through the contours
        for cnt in contours:
            # Calculate the length of the contour
            length = cv.arcLength(cnt, False)
            # Append the contour and its length to the list
            contours_lengths.append((cnt, length))

        # Sort the list of contours by length in descending order
        contours_lengths.sort(key=lambda x: x[1], reverse=True)

        # Select the 5 longest contours
        longest_contours = contours_lengths[:4]
        return longest_contours

    def BottomContourFilter(self, longest_contours, image_y_center):
        Bottom_contours = []
        for cont, _ in longest_contours:

            #Find the Centroid of the contour
            #If the y coordinate of the centroid is greater than half of the image y axis, means that it is a bottom line
            #Between the bottom lines, if there is a min y coordinate value, that is the upper bottom line
            M = cv.moments(cont)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                #print("cx, cy", cx, cy)
                
                if cy > image_y_center:
                    Bottom_contours.append((cont, cy))

        # Sort the list of contours by cY
        Bottom_contours.sort(key=lambda x: x[1])

        #Return the minimum cY which will be the top of the bottom contour
        return Bottom_contours[0][0]

if __name__ == "__main__":
    object_1 = BottomEdgeMapper("imga.BMP")

    object_1.gray_image = object_1.GrayscaleConverter(object_1.image)

    #Define Kernel for Morphological Operations
    object_1.kernel = np.ones((3,3),np.uint8)
    
    #Tried various combination of Morphological operation and kernal size remove noise
    object_1.opening = object_1.MorphologicalOperation_Opening(object_1.gray_image, object_1.kernel)
    object_1.closing = object_1.MorphologicalOperation_Closing(object_1.opening, object_1.kernel)
    object_1.dialation = object_1.MorphologicalOperation_Dialate(object_1.closing, object_1.kernel, iterations = 2)
    object_1.erosion = object_1.MorphologicalOperation_Erode(object_1.dialation, object_1.kernel, iterations = 2)

    #Denoising the image
    object_1.denoised_image = object_1.DeNoiser(object_1.erosion)

    # Thresholding the image. Used trackbars to get to best possible threshold value
    # Suitable to draw contour considering all 4 images provided
    object_1.threshold = object_1.Threshold_Image(object_1.denoised_image, 140, 255)

    # Fine tuning the image using dilation and erosion to join the broken lines
    object_1.dialation = object_1.MorphologicalOperation_Dialate(object_1.threshold, object_1.kernel, iterations = 8)
    object_1.erosion = object_1.MorphologicalOperation_Erode(object_1.dialation, object_1.kernel, iterations = 8)

    # edge detection using canny edge detector
    object_1.edges = object_1.CannyEdgeDetector(object_1.erosion, threshold1=150, threshold2=250, apertureSize=5, L2gradient=True)

    #cv.imshow("object_1.edges", object_1.edges)

    object_1.contours = object_1.Contours(object_1.edges)

    print("contours: ", len(object_1.contours))

    
    object_1.longest_contours = object_1.LongestContoursFilter(object_1.contours)

    object_1.Bottom_contour = object_1.BottomContourFilter(object_1.longest_contours, object_1.image.shape[1]/2)

    # drawing contour on original image
    cv.drawContours(object_1.image_copy, [object_1.Bottom_contour], -1, (0, 255, 0), 2)
    

                 
    # see the results
    cv.imshow("image", object_1.image_copy)

    cv.waitKey(0)
    cv.destroyAllWindows()
