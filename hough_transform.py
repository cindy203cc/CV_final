import cv2
import os
import numpy as np

if __name__ == '__main__':
    img = cv2.imread(os.path.join('datasets', 'solidWhiteRight.jpg'))
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale_img, 100, 200)
    # arguements: edge image, rho and theta accuracy, threshold to be seen as a line
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
    
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join('results', 'solidWhiteRight_result.jpg'), img)