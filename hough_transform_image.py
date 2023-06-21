import cv2
import os
import numpy as np
import copy
import time

'''
crop the ROI region from the image
'''
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    cv2.fillPoly(mask, vertices, [255,255,255])
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image,mask

'''
the vertices of ROI region
'''
def get_vertices(image):
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.05, rows]
    top_left     = [cols*0.45, rows*0.6]
    bottom_right = [cols*0.95, rows]
    top_right    = [cols*0.55, rows*0.6] 
    
    ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return ver

def draw_line(line,img):
    #print('line',line)
    rho = line[0]
    theta = line[1]
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
    start_time = time.time()
    img = cv2.imread(os.path.join('datasets', 'yellowCurve.jpg'))
    img_copy = copy.deepcopy(img)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale_img, 100, 200)
    edges,mask = region_of_interest(edges,get_vertices(edges))
    # arguements: edge image, rho and theta accuracy, threshold to be seen as a line
    lines = cv2.HoughLines(edges, 1, np.pi / 180,threshold = 70)

    right_line = []
    left_line = []
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        #print(line)
        #draw_line(line[0],img)
        if theta < np.pi/2.0 - 0.1:
            right_line.append((rho,theta))
        elif theta > np.pi/2.0 + 0.1:
            left_line.append((rho,theta))

    line_r = np.mean(right_line, axis=0)
    line_l = np.mean(left_line, axis=0)

    img = draw_line(line_r,img)
    img = draw_line(line_l,img)

    foreground = cv2.bitwise_and(img, img, mask=mask)
    background = cv2.bitwise_and(img_copy, img_copy, mask=cv2.bitwise_not(mask))
    img = cv2.bitwise_or(foreground, background)

    end_time = time.time()
    print('Spend {:.3f} second.'.format(end_time - start_time))
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join('results', 'yellowCurve.jpg'), img)