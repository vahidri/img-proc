##NOTE: run with the following command:
## python canny.py input.png

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys


def neighborer(strong, weak):
    data = strong.copy()
    it = 0
    print('>>> running neighborer')
    while True:
        it +=1
        count = 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if 0 == data[i, j]:
                    if weak[i, j]:
                        padded = np.pad(data, 1, 'constant', constant_values=0)
                        vicinity = padded[i:i+3, j:j+3]
                        vicinity[1, 1] = 0
                        vicinity = vicinity.flatten()
                        if vicinity.sum() > 0:
                            data[i, j] = weak[i, j]
                            count += 1
        print(it, '. ', count, sep='')
        if 0 == count:
            break
    return data


def cannyEdge(img , sigma, th1, th2):
    """
    function finds the edges using Canny edge detection method...
    :param img:input image
    :param sigma: sigma is the std-deviation and refers to spread of gaussian
    :param th1:low threshold used to identify weak edges...
    :param th2: high threshold used to identify strong edges...
    :return:
    a binary edge image...
    """

    size = int(2 * (np.ceil(3 * sigma)) + 1)
    x, y = np.meshgrid(np.arange(-size / 2 + 1, size / 2 + 1), np.arange(-size / 2 + 1, size / 2 + 1))
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2)) / normal  # calculating gaussian filter
    kern_size, gauss = kernel.shape[0], np.zeros_like(img, dtype=float)

    for i in range(img.shape[0] - (kern_size - 1)):
        for j in range(img.shape[1] - (kern_size - 1)):
            window = img[i:i + kern_size, j:j + kern_size] * kernel
            gauss[i, j] = np.sum(window)

    kernel, kern_size = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]), 3  # edge detection
    gx = np.zeros_like(gauss, dtype=float)
    gy = np.zeros_like(gauss, dtype=float)

    for i in range(gauss.shape[0] - (kern_size - 1)):
        for j in range(gauss.shape[1] - (kern_size - 1)):
            window = gauss[i:i + kern_size, j:j + kern_size]
            gx[i, j], gy[i, j] = np.sum(window * kernel.T), np.sum(window * kernel)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    theta = ((np.arctan(gy / gx)) / np.pi) * 180  # radian to degree conversion
    nms = np.copy(magnitude)

    theta[theta < 0] += 180

    # non maximum suppression; quantization and suppression done in same step
    for i in range(theta.shape[0] - (kern_size - 1)):
        for j in range(theta.shape[1] - (kern_size - 1)):
            if (theta[i, j] <= 22.5 or theta[i, j] > 157.5):
                if (magnitude[i, j] <= magnitude[i - 1, j]) and (magnitude[i, j] <= magnitude[i + 1, j]): nms[i, j] = 0
            if (theta[i, j] > 22.5 and theta[i, j] <= 67.5):
                if (magnitude[i, j] <= magnitude[i - 1, j - 1]) and (magnitude[i, j] <= magnitude[i + 1, j + 1]): nms[
                    i, j] = 0
            if (theta[i, j] > 67.5 and theta[i, j] <= 112.5):
                if (magnitude[i, j] <= magnitude[i + 1, j + 1]) and (magnitude[i, j] <= magnitude[i - 1, j - 1]): nms[
                    i, j] = 0
            if (theta[i, j] > 112.5 and theta[i, j] <= 157.5):
                if (magnitude[i, j] <= magnitude[i + 1, j - 1]) and (magnitude[i, j] <= magnitude[i - 1, j + 1]): nms[
                    i, j] = 0

    weak, strong = np.copy(nms), np.copy(nms)

    # weak edges
    weak[weak < th1] = 0
    weak[weak > th2] = 0

    # strong edges
    strong[strong < th2] = 0
    strong[strong > th2] = 1

    d = neighborer(strong, weak)
    data = np.zeros((d.shape[0], d.shape[1]), dtype=np.uint8)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if d[i, j]:
                data[i, j] = 255
    aks = Image.fromarray(data, 'L')
    aks.show()
    aks.save('canny-final.png', 'PNG')

    # plotting multiple images
    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    imgplot = plt.imshow(gauss, cmap='gray')
    a.set_title('Gaussian')
    a = fig.add_subplot(2, 2, 2)
    imgplot = plt.imshow(magnitude, cmap='gray')
    a.set_title('Magnitude')
    a = fig.add_subplot(2, 2, 3)
    imgplot = plt.imshow(weak, cmap='gray')
    a.set_title('Weak edges')
    a = fig.add_subplot(2, 2, 4)
    imgplot = plt.imshow(255 - strong, cmap='gray')
    a.set_title('Strong edges')
    #plt.show()
    fig.savefig('canny-plot.png')

if __name__ == '__main__':
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        print("No input image given!")
        quit()
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image=gray, threshold1=90, threshold2=180)
    cv2.imshow("canny - OpenCV", edges)
    cannyEdge(gray, .5, 90, 180)
    cv2.waitKey(0)
