import cv2
import os
import numpy as np

def interest_region(edges):
    h, w = edges.shape
    roi =  np.array([(0, h), (w // 2, h // 2), (w, h)], np.int32)
    dst = np.zeros((edges.shape), np.uint8)
    cv2.fillPoly(dst, [roi], 255)
    return cv2.bitwise_and(edges, dst)

# select the mean parameters of a right line and a left line
def mean_left_and_right(lines):
    left_line = []
    right_line = []
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        #print(line)
        #draw_line(line[0],img)
        if theta < np.pi/2.0:
            right_line.append((rho,theta))
        else:
            left_line.append((rho,theta))

    line_r = np.mean(right_line, axis=0)
    line_l = np.mean(left_line, axis=0)
    return [[line_r], [line_l]]

def plot_lines(img, lines):
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * a)
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * a)
        cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
    return img

if __name__ == '__main__':
    img = cv2.imread(os.path.join('datasets', 'solidWhiteRight.jpg'))
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale_img, 100, 200)
    roi_edges = interest_region(edges)
    # arguements: edge image, rho and theta accuracy, threshold to be seen as a line
    lines = cv2.HoughLines(roi_edges, 1, np.pi / 180, 70)
    lines = mean_left_and_right(lines)
    img = plot_lines(img, lines)
    
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join('results', 'solidWhiteRight_result.jpg'), img)