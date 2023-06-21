'''
流程:
1. 高斯模糊 fin
2. 轉灰階 fin
3. 轉鳥瞰 fin
4. 轉黑白 fin
5. 找左右兩邊的白色點  fin (看起來沒問題拉應該，不過如果找不到第一個高度上的點的畫高機率全都找不到，然後都是迴圈效率很差但我不知道怎麼矩陣化)
6. 用5的點(如果一邊有3個點以上)建構 y = ax^2+bx+c 的matrix去解並得到曲線 fin
7. 把線畫上去然後轉回飛鳥瞰的視角然後把線貼上元照片
'''
import cv2
import os
import numpy as np
import copy
from scipy.optimize import curve_fit
import time
import warnings
from tqdm import tqdm

'''
return the vertices of ROI
'''
def get_vertices(image):
    return np.float32([[200, 720],[1100, 720],[595, 450],[685, 450]])

# 轉成鳥瞰視角
def overhead(transform_h,transform_w,img):
    source = np.float32(get_vertices(img))
    destination = np.float32([[300,720],[980,720],[300,0],[900,0]])
    overhead_transform = cv2.getPerspectiveTransform(source, destination)
    overhead_img = cv2.warpPerspective(img, overhead_transform, dsize=(transform_w, transform_h),flags=cv2.INTER_LINEAR)

    return overhead_img

'''
Unwarped the image from overhead to original perspective
'''
def unoverhead(transform_h,transform_w,img):
    source = np.float32(get_vertices(img))
    destination = np.float32([[300,720],[980,720],[300,0],[900,0]])
    overhead_transform = cv2.getPerspectiveTransform(destination,source)
    overhead_img = cv2.warpPerspective(img, overhead_transform, dsize=(transform_w, transform_h),flags=cv2.INTER_LINEAR)

    return overhead_img

'''
binarize the image to black and white to separate the lane from road
'''
def binary(img):
    sort = np.sort(img.flatten())
    mid = sort[[int(720*1280*0.5)]]
    top = sort[[int(720*1280*0.9)]]
    thres = int(mid+top/2)
    binary_img = cv2.threshold(img,thres,255,cv2.THRESH_BINARY)[1]

    return binary_img

# 用類似window的方式找白色(白色等於在Lane上)
def find_line_points(img):
    win_h = int(img.shape[0]/10) # window高度
    win_w = 200 # window寬度
    half_range = int(win_w/2)

    flag_left = 0 # flag == 0 :還沒找到白色 ; flag == 1 找到第一個白色 ; flag == 2 找到最後一個白色並算出中間點
    flag_right = 0

    left_1 = 0 # 左線第一個白色
    left_2 = 0 # 左線最後一個白色
    right_1 = 0
    right_2 = 0

    left_mid = 333 # left_1 left_2 他倆中間
    right_mid = 1013

    left_points = [] # 左邊的點的集合 ; 0 代表沒有找到白色點，其他值代表該點x座標
    right_points = []

    # 跑最底下第一條線(第一個高度)找到左右邊第一個點
    for i in range(640):
        if img[720-win_h][i] == 255 and flag_left == 0:
            flag_left = 1
            left_1 = i
        elif img[720-win_h][i] == 0 and flag_left == 1:
            flag_left = 2
            left_2 = i
        
        if img[720-win_h][1279-i] == 255 and flag_right == 0:
            flag_right = 1
            right_1 = i
        elif img[720-win_h][1279-i] == 0 and flag_right == 1:
            flag_right = 2
            right_2 = i
        
        if flag_right == 2 and flag_left == 2:break

    if flag_left == 2:
        left_mid = left_1 + int((left_2-left_1)/2)
        left_points.append(left_mid)
    else:
        left_points.append(0)
    if flag_right == 2:
        right_mid = 1280 - (right_2 + int((right_1-right_2)/2))
        right_points.append(right_mid)
    else:
        right_points.append(0)

    flag_left = 0
    flag_right = 0

    # 跑其他8個高度
    for i in range(8):
        for j in range(win_w):
            current_L = img[720-win_h*(i+2)][left_mid-half_range+j]
            current_R = img[720-win_h*(i+2)][right_mid-half_range+j]

            if current_L == 255 and flag_left == 0:
                flag_left = 1
                left_1 = j
            elif current_L == 0 and flag_left == 1:
                flag_left = 2
                left_2 = j
            
            if current_R == 255 and flag_right == 0:
                flag_right = 1
                right_1 = j
            elif current_R == 0 and flag_right == 1:
                flag_right = 2
                right_2 = j

            if right_mid > 1280 - half_range - 5:right_mid = 1280 - half_range - 5

        if flag_left == 2:
            left_mid = left_mid - half_range + left_1 + int((left_2-left_1)/2)
            left_points.append(left_mid)
        else:
            left_points.append(0)
        if flag_right == 2:
            right_mid = right_mid - half_range + right_2 + int((right_1-right_2)/2)
            right_points.append(right_mid)
        else:
            right_points.append(0)

        flag_left = 0
        flag_right = 0

    return left_points,right_points

'''
the formula of the curve
'''
def func(x, a, b, c):
    return a * x**2 + b * x + c

'''
find the parameters of the curve of given edge points
'''
def fit_curve(points):
    xdata = []
    ydata = []
    flag = 0
    last_1 = [0,0]
    last_2 = [0,0]
    for i in range(9):
        if points[i] != 0:
            xdata.append(points[i])
            ydata.append(720 - int(720 * (i+1) / 10))
            if flag == 0:
                last_1 = [720 - int(720 * (i+1) / 10),points[i]]
                flag = 1
            else:
                last_2 = last_1
                last_1 = [720 - int(720 * (i+1) / 10),points[i]]
                flag = 2
    # fake point
    if flag == 2:
        fake_x = int((last_1[0]*last_2[1] - last_2[0]*last_1[1])/(last_1[0] - last_2[0]))
        xdata.append(fake_x)
        ydata.append(0)
        delta_x = xdata[0] - xdata[1]
        delta_y = ydata[0] - ydata[1]
        fake_x = int((800 - ydata[0]) * (delta_x / delta_y) + xdata[0])
        xdata.insert(0, fake_x)
        ydata.insert(0, 800 - 1)
        popt, _ = curve_fit(func, xdata, ydata)
        return popt,xdata
    else:
        return 0,[]

'''
plot the lane line on the binary image
'''
def generate_line_image(binary_img):
    line_img = np.zeros((720,1280), np.uint8)

    left_points,right_points = find_line_points(binary_img)
    left_popt,xdata = fit_curve(left_points)
    xlimit = 0
    xmin = 0
    if len(xdata) >= 3:
        xlimit = np.max(xdata)
        xmin = np.min(xdata)
        left_x = np.linspace(xmin, xlimit, 100)
        left_y = func(left_x, *left_popt)
        cv2.polylines(line_img, pts=[np.array([*zip(left_x, left_y)], np.int32)], isClosed=False, color=255, thickness=15)

    right_popt,xdata = fit_curve(right_points)
    if len(xdata) >= 3:
        xlimit = np.max(xdata)
        xmin = np.min(xdata)
        right_x = np.linspace(xmin, xlimit, 100)
        right_y = func(right_x, *right_popt)
        cv2.polylines(line_img, pts=[np.array([*zip(right_x, right_y)], np.int32)], isClosed=False, color=255, thickness=15)

    return line_img

def draw_line(img):
    img = cv2.resize(img,(1280,720), interpolation=cv2.INTER_AREA)
    img_copy = copy.deepcopy(img)
    img = cv2.GaussianBlur(img, (5,5), 0)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    overhead_img = overhead(720,1280,grayscale_img)
    binary_img = binary(overhead_img)

    line_img = generate_line_image(binary_img)

    line_img = unoverhead(720,1280,line_img)
    foreground = np.zeros((img_copy.shape), np.uint8)
    foreground[:, :, 2] = line_img
    background = cv2.bitwise_and(img_copy, img_copy, mask=cv2.bitwise_not(line_img))
    img = cv2.bitwise_or(foreground, background)

    #cv2.imshow('img',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite(os.path.join('results', 'solidWhiteRight_result.jpg'), img)

    return img

if __name__ == '__main__':
    #img = cv2.imread(os.path.join('datasets', 'solidWhiteRight.jpg'))
    #draw_line(img)
    cap = cv2.VideoCapture(os.path.join('datasets', 'project_video.mp4'))
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(os.path.join('results', 'project_video_result.mp4'), fourcc, fps, (w, h))
    
    start_time = time.time()
    for i in tqdm(range(video_length)):
        if cap.isOpened():
            ret, img = cap.read()
        if not ret:
            break
        img = draw_line(img)
        out.write(img)

    end_time = time.time()
    print('Spend {:.3f} second.'.format(end_time - start_time))
    cap.release()
    out.release()