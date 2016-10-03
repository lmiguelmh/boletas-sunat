# @author lmiguelmh
# @since 20161002T1741

import numpy as np
import cv2


def getROI(image, minArea=300 * 200, windowSize=31, corners=4, countCorners=True, adaptiveThresholding=True):
    """
    detect full 4-corner object, that is the 4 corners must be visible
    todo: detect no 4-borders objects for example: use line detectors in conjunction with contours

    :param image:
    :param minArea:
    :param windowSize:
    :param corners:
    :param countCorners:
    :param adaptiveThresholding:
    :return:
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, int(windowSize / 5), int(255 / 16), windowSize)
    if adaptiveThresholding:
        bin = cv2.adaptiveThreshold(blur, 255, cv2.THRESH_BINARY_INV, cv2.ADAPTIVE_THRESH_MEAN_C, windowSize, 0)
    else:
        _, bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _image, contours, hierarchy = cv2.findContours(bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        raise Exception("no contours detected: adjust brightness and borders")

    roicnt = sorted(contours, key=cv2.contourArea, reverse=True)[:1][0]
    roi = None
    # final = np.full(image.shape, 255, dtype=np.uint8)
    if cv2.contourArea(roicnt) > minArea:
        approx = cv2.approxPolyDP(roicnt, 0.015 * cv2.arcLength(roicnt, True), True)
        if not countCorners or len(approx) == corners:
            points = np.asarray([approx[0][0], approx[1][0], approx[2][0], approx[3][0]], dtype=np.float32)
            sum = np.sum(points, axis=1)
            rect = np.full((4, 2), 0, dtype=np.float32)
            rect[0] = points[np.argmin(sum)]  # topLeft
            rect[2] = points[np.argmax(sum)]  # bottomRight
            diff = np.diff(points, axis=1)
            rect[1] = points[np.argmin(diff)]  # topRight
            rect[3] = points[np.argmax(diff)]  # bottomLeft
            (topl, topr, bottomr, bottoml) = rect
            # calc distance
            bottomWidth = np.sqrt(((bottomr[0] - bottoml[0]) ** 2) + ((bottomr[1] - bottoml[1]) ** 2))
            topWidth = np.sqrt(((topr[0] - topl[0]) ** 2) + ((topr[1] - topl[1]) ** 2))
            rightHeight = np.sqrt(((topr[0] - bottomr[0]) ** 2) + ((topr[1] - bottomr[1]) ** 2))
            leftHeight = np.sqrt(((topl[0] - bottoml[0]) ** 2) + ((topl[1] - bottoml[1]) ** 2))
            maxWidth = max(int(bottomWidth), int(topWidth))
            maxHeight = max(int(rightHeight), int(leftHeight))
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            roi = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            # cv2.drawContours(final, approx, -1, color=(0, 0, 0), thickness=10)
        else:
            raise Exception("only 4-corner objects for now")  # todo use line detection and borders
    else:
        raise Exception("no roi detected: the object must be the only 4-corner object in the image")

    return roi


def getROIHeader(roi):
    """
    roi is the 4-corner object
    :param roi:
    :return:
    """
    w = roi.shape[0]
    h = roi.shape[1]
    wc = int(w / 2)
    hc = int(h / 2)
    q1 = roi[0:wc, 0:hc]
    q1roi = None
    try:
        q1roi = getROI(q1, w * h / 16, adaptiveThresholding=False)  # 16th part of roi
    except Exception as e:
        print("q1", e)

    q2 = roi[wc:w, 0:hc]
    q2roi = None
    try:
        q2roi = getROI(q2, w * h / 16, adaptiveThresholding=False)  # 16th part of roi
    except Exception as e:
        print("q2", e)

    q3 = roi[wc:w, hc:h]
    q3roi = None
    try:
        q3roi = getROI(q3, w * h / 16, adaptiveThresholding=False)  # 16th part of roi
    except Exception as e:
        print("q3", e)

    q4 = roi[0:wc, hc:h]
    q4roi = None
    try:
        q4roi = getROI(q4, w * h / 16, adaptiveThresholding=False)  # 16th part of roi
    except Exception as e:
        print("q4", e)

    roiheader = None
    if q1roi is not None:
        if roiheader is None:
            roiheader = q1roi
        else:
            raise Exception("more than one header found: unsupported or unknow object")
    if q2roi is not None:
        if roiheader is None:
            roiheader = q2roi
        else:
            raise Exception("more than one header found: unsupported or unknow object")
    if q3roi is not None:
        if roiheader is None:
            roiheader = q3roi
        else:
            raise Exception("more than one header found: unsupported or unknow object")
    if q4roi is not None:
        if roiheader is None:
            roiheader = q4roi
        else:
            raise Exception("more than one header found: unsupported or unknow object")
    return roiheader