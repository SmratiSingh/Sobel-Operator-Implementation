
from skimage import io
from skimage.color import rgb2gray
import numpy as np
import math
import matplotlib.pyplot as mplib
import cv2

#Reading image


#Converting image to gray-scale
#image_gray = rgb2gray(image)

#Displaying image
# mplib.imshow(image)
# mplib.show()

# print(image.shape)

sobelFilterX = np.zeros(shape=(3, 3))
sobelFilterX[0, 0] = 1
sobelFilterX[0, 1] = 0
sobelFilterX[0, 2] = -1
sobelFilterX[1, 0] = 2
sobelFilterX[1, 1] = 0
sobelFilterX[1, 2] = -2
sobelFilterX[2, 0] = 1
sobelFilterX[2, 1] = 0
sobelFilterX[2, 2] = -1
# print(sobelFilterX)


sobelFilterY = np.zeros(shape=(3, 3))
sobelFilterY[0, 0] = 1
sobelFilterY[0, 1] = 2
sobelFilterY[0, 2] = 1
sobelFilterY[1, 0] = 0
sobelFilterY[1, 1] = 0
sobelFilterY[1, 2] = 0
sobelFilterY[2, 0] = -1
sobelFilterY[2, 1] = -2
sobelFilterY[2, 2] = -1
print(np.array(sobelFilterY))


#Convoluting the Sobel filter matrix with Image-subpart
def filterSobel(imgSub, filtr):
    newval = 0
    for i in range(0, 3):
        for j in range(0, 3):
            newval = newval + (imgSub[i, j] * filtr[i, j])
    return newval


#Taking sub-parts of image
def imageSubPart(img, rows, cols):
    subImg = np.zeros(shape=(3, 3))
    r = rows
    c = cols

    for i in range(0, 3):
        c = cols
        for j in range(0, 3):
            subImg[i, j] = img[r, c]
            c = c+1
        r = r+1
    return subImg


#Applying Vertical Sobel Filter
def XFilter(image1):
    r = image1.shape[0]
    c = image1.shape[1]
    finalImage = np.zeros(shape=(r, c))
    subImg = np.zeros(shape=(3, 3))
    #Taking sub-parts of main image one by one
    for i in range(0, r-2):
        for j in range(0, c-2):
            #Calling func. to apply Sobel filter to the image sub-part
            subImg = imageSubPart(image1, i, j)
            newval = filterSobel(subImg, sobelFilterX)
            if (newval < 150):
                newval = 0

            else:
                newval = 1
            finalImage[i, j] = newval
    return finalImage


#Applying Horizontal Sobel Filter
def YFilter(image1):
    r = image1.shape[0]
    c = image1.shape[1]
    finalImage = np.zeros(shape=(r, c))
    subImg = np.zeros(shape=(3, 3))
    # Taking sub-parts of main image one by one
    for i in range(0, r - 2):
        for j in range(0, c - 2):
            # Calling func. to apply Sobel filter to the image sub-part
            subImg = imageSubPart(image1, i, j)
            newval = filterSobel(subImg, sobelFilterY)

            if(newval<150):
                newval = 0

            else:
                newval = 1
            finalImage[i, j] = newval
    return finalImage


def gradMagni(gradx, grady):
    row = gradX.shape[0]
    col = gradY.shape[1]
    g = np.zeros(shape=(row, col))
    for i in range(0, row):
        for j in range(0, col):
            g[i, j] = math.sqrt((gradX[i, j]**2) + (gradY[i, j]**2))
    return g


image = cv2.imread('input.jpg', 0)


# mplib.imshow(image)
# mplib.show()

gradX = XFilter(image)
gradY = YFilter(image)
finalImage = gradMagni(gradX, gradY)
print(gradY.shape)
print(gradX.shape)


cv2.imshow("image", image)
cv2.imshow("imgX", gradX)
cv2.imshow("imgY", gradY)
cv2.imshow("magnitude", finalImage)

cv2.waitKey(0)
cv2.destroyAllWindows()


# mplib.imshow(gradX)
# mplib.show()
#
# mplib.imshow(gradY)
# mplib.show()




