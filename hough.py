##NOTE: run with the following command:
## python hough.py input.png 
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys


def hough_transform(img_bin, theta_res=1, rho_res=1):
    '''
    calculation Hough transform using polar coordinates, scanning thru input image looking for edges; intersecting and so on
    :param img_bin:
    :param theta_res:
    :param rho_res:
    :return:
    '''
    nR, nC = img_bin.shape
    theta = np.linspace(-90.0, 0.0, np.ceil(90.0 / theta_res) + 1.0)
    theta = np.concatenate((theta, -theta[len(theta) - 2::-1]))

    D = np.sqrt((nR - 1) ** 2 + (nC - 1) ** 2)
    q = np.ceil(D / rho_res)
    nrho = 2 * q + 1
    rho = np.linspace(-q * rho_res, q * rho_res, nrho)
    H = np.zeros((len(rho), len(theta)))
    for rowIdx in range(nR):
        for colIdx in range(nC):
            if img_bin[rowIdx, colIdx]:
                for thIdx in range(len(theta)):
                    rhoVal = colIdx * np.cos(theta[thIdx] * np.pi / 180.0) + \
                             rowIdx * np.sin(theta[thIdx] * np.pi / 180)
                    rhoIdx = np.nonzero(np.abs(rho - rhoVal) == np.min(np.abs(rho - rhoVal)))[0]
                    H[rhoIdx[0], thIdx] += 1
    return rho, theta, H


def top_n_rho_theta_pairs(ht_acc_matrix, n, rhos, thetas):
    '''
  @param hough transform accumulator matrix H (rho by theta)
  @param n pairs of rho and thetas desired
  @param ordered array of rhos represented by rows in H
  @param ordered array of thetas represented by columns in H
  @return top n rho theta pairs in H by accumulator value
  @return x,y indexes in H of top n rho theta pairs
  '''
    flat = list(set(np.hstack(ht_acc_matrix)))
    flat_sorted = sorted(flat, key=lambda n: -n)
    coords_sorted = [(np.argwhere(ht_acc_matrix == acc_value)) for acc_value in flat_sorted[0:n]]
    rho_theta = []
    x_y = []
    for coords_for_val_idx in range(0, len(coords_sorted), 1):
        coords_for_val = coords_sorted[coords_for_val_idx]
        for i in range(0, len(coords_for_val), 1):
            n, m = coords_for_val[i]
            rho = rhos[n]
            theta = thetas[m]
            rho_theta.append([rho, theta])
            x_y.append([m, n])
    return [rho_theta[0:n], x_y]


def valid_point(pt, ymax, xmax):
    '''
  @return True/False if pt is with bounds for an xmax by ymax image
  '''
    x, y = pt
    if x <= xmax and x >= 0 and y <= ymax and y >= 0:
        return True
    else:
        return False


def round_tup(tup):
    '''
  @return closest integer for each number in a point for referencing
  a particular pixel in an image
  '''
    x, y = [int(round(num)) for num in tup]
    return (x, y)


def draw_rho_theta_pairs(target_im, pairs):
    '''
  @param opencv image
  @param array of rho and theta pairs
  Has the side-effect of drawing a line corresponding to a rho theta
  pair on the image provided
  '''
    im_y_max, im_x_max, channels = np.shape(target_im)
    for i in range(0, len(pairs), 1):
        point = pairs[i]
        rho = point[0]
        theta = point[1] * np.pi / 180  # degrees to radians
        # y = mx + b form
        m = -np.cos(theta) / np.sin(theta)
        b = rho / np.sin(theta)
        # possible intersections on image edges
        left = (0, b)
        right = (im_x_max, im_x_max * m + b)
        top = (-b / m, 0)
        bottom = ((im_y_max - b) / m, im_y_max)

        pts = [pt for pt in [left, right, top, bottom] if valid_point(pt, im_y_max, im_x_max)]
        if len(pts) == 2:
            cv2.line(target_im, round_tup(pts[0]), round_tup(pts[1]), (0, 0, 255), 1)


if __name__ == '__main__':

    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        print("No input image given!")
        quit()

    img_orig = cv2.imread(filename, )
    img = img_orig[:, :, ::-1]  # color channel plotting mess http://stackoverflow.com/a/15074748/2256243

    print('0')

    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('.5 color')
    edges = cv2.Canny(bw, threshold1=0, threshold2=50, apertureSize=3)
    print('.7 canny')
    print('THIS COULD TAKE A LONG TIME...')
    rhos, thetas, H = hough_transform(edges)
    print('1 hough')
    rho_theta_pairs, x_y_pairs = top_n_rho_theta_pairs(H, 22, rhos, thetas)
    im_w_lines = img.copy()
    draw_rho_theta_pairs(im_w_lines, rho_theta_pairs)
    print('2')

    for i in range(0, len(x_y_pairs), 1):
        x, y = x_y_pairs[i]
        cv2.circle(img=H, center=(x, y), radius=12, color=(0, 0, 0), thickness=1)
    print('3')

    plt.subplot(141), plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(edges, cmap='gray')
    plt.title('Image Edges'), plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(H)
    plt.title('Hough Transform Accumulator'), plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(im_w_lines)
    plt.title('Detected Lines'), plt.xticks([]), plt.yticks([])
    plt.show()

