import cv2
import numpy as np

ctr=0

def filter_red(frame):
    global ctr
    ctr+=1
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([20, 255, 255])

    lower_red2 = np.array([150, 100, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask1 = mask1 + mask2

    final = cv2.bitwise_and(frame, frame, mask=mask1)
    final[np.where((final == [0]).all(axis=2))] = [255]

    cv2.imshow('final', final)
    k= cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    if ctr < 4:
        cv2.imwrite('red1.jpg', final)
    else:
        cv2.imwrite('red2.jpg', final)

def filter_green(frame):
    global ctr
    ctr+=1
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([30, 100, 50])
    upper_green = np.array([90, 255, 255])

    mask1 = cv2.inRange(hsv, lower_green, upper_green)

    final = cv2.bitwise_and(frame, frame, mask=mask1)
    final[np.where((final == [0]).all(axis=2))] = [255]

    cv2.imshow('final', final)
    k= cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    if ctr < 4:
        cv2.imwrite('green1.jpg', final)
    else:
        cv2.imwrite('green2.jpg', final)

def filter_blue(frame):
    global ctr
    ctr+=1
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 0])
    upper_blue = np.array([150, 255, 255])

    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)

    final = cv2.bitwise_and(frame, frame, mask=mask1)
    final[np.where((final == [0,0,0]).all(axis=2))] = [255]


    cv2.imshow('final', final)
    k= cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    if ctr<4:
        cv2.imwrite('blue1.jpg', final)
    else:
        cv2.imwrite('blue2.jpg', final)

img1 = cv2.imread('opencv1.jpeg')
filter_red(img1)
filter_green(img1)
filter_blue(img1)

img2 = cv2.imread('opencv2.png')
filter_red(img2)
filter_green(img2)
filter_blue(img2)








import cv2
import numpy as np

if __name__ == '__main__':
    # Read image
    im = cv2.imread("asgmt5_img.jpg")

    # Select ROI
    r = cv2.selectROI(im)
    print(r[0],"   ",r[1],"   ",r[2],"   ",r[3])

    # Crop image
    imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

    # Display cropped image
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0)
