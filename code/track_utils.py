import argparse
import cv2
from singleton import *

# This file contains useful function to detect and track the ball and the players

# Setting some variables for the script

img = None
orig = None

# Define the Region of Interest
roi = None
ix, iy = 0, 0

# Define the bool draw
draw = False


def getArguments():
    """
    Get the input video parameter or camera for execution
    :return: args
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    args = vars(ap.parse_args())
    return args


def resize_video(img, width=400.0):
    """
    Reside the video according to the image ratio
    :param img: the image
    :param width: the width specified
    :return: img_resize
    """
    r = float(width) / img.shape[0]
    dim = (int(img.shape[1] * r), int(width))
    img_resize = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img_resize


def selectROI(event, x, y, flag, param):
    """
    Select the ROI (Region of Interest)
    :param event: the event that is happened (click, select, ..)
    :param x: x coordinate
    :param y: y coordinate
    """

    # In this function I select the ROI (Region of Interest)
    global ix, iy, draw, img, orig, roi
    # check if the left mouse button is pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        ix = x
        iy = y
        draw = True

    # check if the mouse pointer has moved over the window
    elif event == cv2.EVENT_MOUSEMOVE:
        if draw:
            # draw a rectangle
            img = cv2.rectangle(orig.copy(), (ix, iy), (x, y), (255, 0, 0), 2)

    # check if left mouse button is released
    elif event == cv2.EVENT_LBUTTONUP:
        if draw:
            x1 = max(x, ix)
            y1 = max(y, iy)
            ix = min(x, ix)
            iy = min(y, iy)
            roi = orig[iy:y1, ix:x1]
        draw = False


def getROIvid(frame, winName='input'):
    """
    Get the input of the ROI
    :param frame: frame of the video
    :param winName: the name of the window (input in this case)
    :return: roi (region of interest)
    """
    global img, orig, roi
    roi = None
    img = frame.copy()
    orig = frame.copy()
    cv2.namedWindow(winName)
    cv2.setMouseCallback(winName, selectROI)
    while True:
        cv2.imshow(winName, img)
        if roi is not None:
            cv2.destroyWindow(winName)
            return roi

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            cv2.destroyWindow(winName)
            break

    return roi


def getROIext(image, winName='input'):
    """
    Get the exit from the ROI
    :param image: image of the windows
    :param winName: name of the window (input in this case)
    :return: roi
    """
    global img, orig
    img = image.copy()
    orig = image.copy()
    cv2.namedWindow(winName)
    cv2.setMouseCallback(winName, selectROI)
    while True:
        cv2.imshow(winName, img)
        if roi is not None:
            cv2.destroyWindow(winName)
            return roi

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            cv2.destroyWindow(winName)
            break

    return roi


def getLimits(roi):
    """
    Get the limits of the region of interest
    :param roi: region of interest
    :return: limits
    """
    limits = None
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(roi)
    limits = [(int(np.amax(h)), int(np.amax(s)), 255), (int(np.amin(h)), int(np.amin(s)), int(np.amin(v)))]
    return limits


def applyMorphTransforms(mask):
    """
    Apply the morph transformation
    :param mask: mask of the ball
    :return: modified mask
    """
    # instantiate class object
    singleton_instance = Singleton()
    lower = 100
    upper = 255

    mask = cv2.GaussianBlur(mask, (11, 11), 5)
    mask = cv2.inRange(mask, lower, upper)
    mask = cv2.dilate(mask, singleton_instance.kernel)
    mask = cv2.erode(mask, np.ones((5, 5)))

    return mask


def applyMorphTransforms2(backProj):
    """
    Apply the morph transformation
    :param backProj: back-projection of the histogram
    :return: mask
    """
    # instantiate class object
    singleton_instance = Singleton()
    lower = 50
    upper = 255
    mask = cv2.inRange(backProj, lower, upper)
    mask = cv2.dilate(mask, singleton_instance.kernel)
    mask = cv2.erode(mask, np.ones((3, 3)))
    mask = cv2.GaussianBlur(mask, (11, 11), 5)
    mask = cv2.inRange(mask, lower, upper)
    return mask


def detectBallThresh(frame, limits):
    """
    Detect the ball using color thresholding
    :param frame: frame of the video
    :param limits: limits of the area
    :return: center and the i-th contour
    """
    singleton_instance = Singleton()
    upper = limits[0]
    lower = limits[1]
    center = None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = applyMorphTransforms(mask)
    cv2.imshow('mask', mask)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    flag = False
    i = 0
    if len(contours) > 0:
        for i in range(len(contours)):
            (_, radius) = cv2.minEnclosingCircle(contours[i])
            if singleton_instance.rad_thresh > radius > 5:
                flag = True
                break
        if not flag:
            return None, None

        M = cv2.moments(contours[i])
        # compute the center of mass of the ball
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return center, contours[i]
    else:
        return None, None


def detectBallHB(frame, roi):
    """
    Detect the ball histogram
    :param frame: frame of the video
    :param roi: region of interest
    :return: center and contours
    """
    singleton_instance = Singleton()
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # convert to HSV colour space

    # calculate the histogram of the roi set
    roiHist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # normalize the norm of the array obtained
    cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # compute the back projection of the histogram
    backProj = cv2.calcBackProject([hsv], [0, 1], roiHist, [0, 180, 0, 256], 1)

    mask = cv2.inRange(backProj, 50, 255)
    # erode the image
    mask = cv2.erode(mask, np.ones((5, 5)))
    # dilate the image
    mask = cv2.dilate(mask, np.ones((5, 5)))

    cv2.imshow('mask', mask)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    i = 0
    if len(contours) > 0:
        for i in range(len(contours)):
            (_, radius) = cv2.minEnclosingCircle(contours[i])
            if singleton_instance.rad_thresh > radius > 5:
                break
        M = cv2.moments(contours[i])
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return center, contours[i]
    else:
        return None, None


def kalmanFilter(meas):
    """
    Apply the kalman filter
    :param meas: measurements
    :return: prediction
    """
    # The tracking uses what is known in literature as ???Kalman Filter???, it is an ???asymptotic state estimator???,
    # a mathematical tool that allows to estimate the position of the tracked object using the cinematic model of
    # the object and its ???history???
    pred = np.array([], dtype=np.int)
    # mp = np.asarray(meas,np.float32).reshape(-1,2,1)  # measurement
    tp = np.zeros((2, 1), np.float32)  # tracked / prediction

    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.00003
    for mp in meas:
        mp = np.asarray(mp, dtype=np.float32).reshape(2, 1)
        kalman.correct(mp)
        tp = kalman.predict()
        np.append(pred, [int(tp[0]), int(tp[1])])

    return pred


def removeBG(frame, background_subtractor):
    bg_mask = background_subtractor.apply(frame)
    bg_mask = cv2.dilate(bg_mask, np.ones((5, 5)))
    frame = cv2.bitwise_and(frame, frame, mask=bg_mask)
    return frame


def getHist(frame):
    """
    Get the histograms of the two teams (A and B)
    :param frame: frame of the video
    :return: roi_hist_A, roi_hist_B
    """
    roi_hist_A, roi_hist_B = None, None

    if roi_hist_A is None:
        roi = getROIvid(frame, 'input team A')
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist_A = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        roi_hist_A = cv2.normalize(roi_hist_A, roi_hist_A, 0, 255, cv2.NORM_MINMAX)

    if roi_hist_B is None:
        roi = getROIvid(frame, 'input team B')
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist_B = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        roi_hist_B = cv2.normalize(roi_hist_B, roi_hist_B, 0, 255, cv2.NORM_MINMAX)

    return roi_hist_A, roi_hist_B
