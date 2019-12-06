##NOTE: run with the following command:
## python cons.py input.png
from pprint import pprint as pp
from skimage.exposure import rescale_intensity
import numpy as np
import cv2
import sys

class Filters:
    smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
    largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
    sharpen = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")
    laplacian = np.array((
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]), dtype="int")

    sobelX = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype="int")

    sobelY = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), dtype="int")

def convolve(image, kernel):
    imageHeight, imageWidth = image.shape[:2]
    kernelHeight, kernelWidth = kernel.shape[:2]
    padding = (kernelWidth - 1) // 2
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    ##initialize output
    output = np.zeros((imageHeight, imageWidth), dtype="float32")
    for y in np.arange(padding, imageHeight + padding):
        for x in np.arange(padding, imageWidth + padding):
            area = image[y - padding:y + padding + 1, x - padding:x + padding + 1]
            k = (area * kernel).sum()
            output[y - padding, x - padding] = k
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output


def sobel(image):
    Gx = convolve(image, Filters.sobelX)
    Gy = convolve(image, Filters.sobelY)
    G = np.zeros(Gx.shape, dtype=np.uint8)
    for i in range(Gx.shape[0]):
        for j in range(Gy.shape[1]):
            G[i][j] = (Gx[i][j] ** 2 + Gy[i][j] ** 2) ** (1 / 2)
    return G, Gx, Gy


if __name__ == '__main__':
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        print("No input image given!")
        quit()

    ## grayscale the image
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for r in gray:
        for e in r:
            print(255-e,end=",")
        print(" ", end="")
    quit()

    kernelList = (
        ("sobel_x", Filters.sobelX),
        ("sobel_y", Filters.sobelY)
    )
    ## process the image according to the kernelList
    cv2.imshow("original", gray)
    for (kernelName, kernel) in kernelList:
        print(">>> applying {} kernel".format(kernelName))
        outConvole = convolve(gray, kernel)
        cv2.imshow("{} - convole".format(kernelName), outConvole)
        #outOpencv = cv2.filter2D(gray, -1, kernel)
        #cv2.imshow("{} - opencv".format(kernelName), outOpencv)
    cv2.imshow("Sobel", sobel(image)[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
