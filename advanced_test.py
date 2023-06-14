'''
流程:
1. 高斯模糊 fin
2. 轉灰階 fin
3. 轉鳥瞰 fin
4. 轉黑白 fin
5. 找左右兩邊的白色點  fin (看起來沒問題拉應該，不過如果找不到第一個高度上的點的畫高機率全都找不到，然後都是迴圈效率很差但我不知道怎麼矩陣化)
6. 用5的點(如果一邊有3個點以上)建構 y = ax^2+bx+c 的matrix去解並得到曲線
7. 把線畫上去然後轉回飛鳥瞰的視角然後把線貼上元照片
'''
import cv2
import os
import numpy as np
import copy
from scipy.optimize import curve_fit

# 回傳要轉的四個角落，改了會炸
def get_vertices(image):
    return np.float32([[200, 720],[1100, 720],[595, 450],[685, 450]])

# 轉成鳥瞰視角
def overhead(transform_h,transform_w,img):
    source = np.float32(get_vertices(img))
    destination = np.float32([[300,720],[980,720],[300,0],[900,0]])
    overhead_transform = cv2.getPerspectiveTransform(source, destination)
    overhead_img = cv2.warpPerspective(img, overhead_transform, dsize=(transform_w, transform_h),flags=cv2.INTER_LINEAR)

    return overhead_img

# 轉黑白二極影像
def binary(img):
    sort = np.sort(overhead_img.flatten())
    mid = sort[[int(720*1280*0.5)]]
    top = sort[[int(720*1280*0.9)]]
    thres = int(mid+top/2)
    binary_img = cv2.threshold(overhead_img,thres,255,cv2.THRESH_BINARY)[1]

    return binary_img

# 用類似window的方式找白色(白色等於在Lane上) (我懶只找了window中間線上的點)
def find_line_points(img):
    win_h = int(img.shape[0]/10) # window高度
    win_w = 100 # window寬度
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

        if right_mid > 1280 - 51:right_mid = 1280 - 51

    return left_points,right_points

def func(x, a, b, c):
    return a * x**2 + b * x + c

def fit_curve(points):
    xdata = []
    ydata = []
    for i in range(9):
        if points[i] != 0:
            xdata.append(points[i])
            ydata.append(720 - int(720 * (i+1) / 10))
    popt, _ = curve_fit(func, xdata, ydata)
    # bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    return popt

if __name__ == '__main__':
    '''
    流程:
    1. 高斯模糊 fin
    2. 轉灰階 fin
    3. 轉鳥瞰 fin
    4. 轉黑白 fin
    5. 找左右兩邊的白色點  fin (看起來沒問題拉應該，不過如果找不到第一個高度上的點的畫高機率全都找不到，然後都是迴圈效率很差但我不知道怎麼矩陣化)
    6. 用5的點(如果一邊有3個點以上)建構 y = ax^2+bx+c 的matrix去解並得到曲線
    7. 把線畫上去然後轉回飛鳥瞰的視角然後把線貼上元照片
    '''
    img = cv2.imread(os.path.join('datasets', 'test_3.jpg'))
    img_copy = copy.deepcopy(img)
    img = cv2.GaussianBlur(img, (5,5), 0)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    overhead_img = overhead(720,1280,grayscale_img)
    binary_img = binary(overhead_img)

    left_points,right_points = find_line_points(binary_img)
    left_popt = fit_curve(left_points)
    x = np.linspace(0, 640, 100)
    y = func(x, *left_popt)
    cv2.polylines(binary_img, pts=[np.array([*zip(x, y)], np.int32)], isClosed=False, color=128, thickness=3)
    print(left_points)
    print(right_points)

    # 用來看找線上白色點的線是在哪
    for j in range(9):
        for i in range(1280):
            binary_img[720-72-72*j,i] = 255
    binary_img[500,1013] = 255

    for i in range(9):
        y = 720 - int(720 * (i+1) / 10)
        cv2.circle(binary_img, (left_points[i], y), 5, 128, -1)
        cv2.circle(binary_img, (right_points[i], y), 5, 128, -1)

    cv2.imshow('img', binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join('results', 'solidWhiteRight_result.jpg'), img)